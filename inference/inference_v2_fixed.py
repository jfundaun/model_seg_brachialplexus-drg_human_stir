#!/usr/bin/env python3
"""
Cascaded nnUNet Inference: DRG + Brachial Plexus Segmentation
==============================================================
Stage 1: Dataset201 (coarse, single-channel T2 input)
Stage 2: Dataset202 (fine 16-class, 2-channel: T2 + coarse seg)

Usage:
    python inference_v2_fixed.py --subset all \\
        --output /scratch/users/jfundaun/bpseg/derivatives/cascade_results_final/19June2026
"""

import os
import argparse
import shutil
import tempfile
import subprocess
import contextlib
import warnings
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.ndimage import center_of_mass

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    warnings.warn("SimpleITK not found -- ASD/HD/HD95 will be NaN.  pip install SimpleITK")
    HAS_SITK = False

# ============================================================
# CONFIGURATION
# ============================================================
NNUNET_BASE    = "/scratch/users/jfundaun/bpseg/nnunet_cascade"
NNUNET_RAW     = f"{NNUNET_BASE}/raw"
NNUNET_RESULTS = f"{NNUNET_BASE}/results"
NNUNET_PREPROC = f"{NNUNET_BASE}/preprocessed"

# Dataset201: single-channel T2 images (used as Stage 1 input)
D201        = f"{NNUNET_RAW}/Dataset201_DRGPlexusCoarse"
D201_IMG_TR = f"{D201}/imagesTr"
D201_IMG_TS = f"{D201}/imagesTs"

# Dataset202: fine 16-class GT labels for evaluation
D202        = f"{NNUNET_RAW}/Dataset202_DRGPlexusFine"
D202_LBL_TR = f"{D202}/labelsTr"
D202_LBL_TS = f"{D202}/labelsTs"

TRAINER       = "nnUNetTrainer"
PLANS         = "nnUNetResEncUNetMPlans"
CONFIGURATION = "3d_fullres"
FOLDS         = [0, 1, 2, 3, 4]   # all 5 folds -- full ensemble

# Placeholder seg path used in the CSV for truly unlabeled cases.
PLACEHOLDER_SEG = "003_032_seg.nii"

FINE_LABEL_NAMES = {
    0:  "background",
    1:  "DRG_C5_R",  2:  "DRG_C6_R",  3:  "DRG_C7_R",  4:  "DRG_C8_R",
    5:  "DRG_C5_L",  6:  "DRG_C6_L",  7:  "DRG_C7_L",  8:  "DRG_C8_L",
    9:  "BP_C5_R",   10: "BP_C6_R",   11: "BP_C7_R",   12: "BP_C8_R",
    13: "BP_C5_L",   14: "BP_C6_L",   15: "BP_C7_L",   16: "BP_C8_L",
}
DRG_CLASSES    = list(range(1, 9))
PLEXUS_CLASSES = list(range(9, 17))


# ============================================================
# ENVIRONMENT
# ============================================================
def set_nnunet_env():
    os.environ["nnUNet_raw"]          = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROC
    os.environ["nnUNet_results"]      = NNUNET_RESULTS


# ============================================================
# FILE HELPERS
# ============================================================
def is_gzip(path):
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


def safe_save_as_nii_gz(src, dst):
    if is_gzip(src):
        shutil.copy2(src, dst)
    else:
        print(f"    [re-gzip] {Path(src).name} -- not real gzip, re-saving...")
        img = nib.load(src)
        nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, img.header), dst)


def case_id_from_image(path):
    name = Path(path).name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[:-len(ext)]
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def find_label(case_id, label_dir):
    for ext in (".nii.gz", ".nii"):
        p = Path(label_dir) / f"{case_id}{ext}"
        if p.exists():
            return str(p)
    return None


def is_placeholder_seg(seg_path):
    return seg_path is None or Path(str(seg_path)).name == PLACEHOLDER_SEG


# ============================================================
# COLLECT CASES  --  original nnUNet-raw-dir mode
# ============================================================
def collect_cases(subset="all"):
    cases = []
    sources = []
    if subset in ("all", "train"):
        sources.append((D201_IMG_TR, D202_LBL_TR, "train"))
    if subset in ("all", "test"):
        sources.append((D201_IMG_TS, D202_LBL_TS, "test"))

    for img_dir, lbl_dir, subset_tag in sources:
        img_dir_p = Path(img_dir)
        if not img_dir_p.exists():
            print(f"  WARNING: image directory not found: {img_dir}")
            continue
        images = sorted(img_dir_p.glob("*_0000.nii.gz")) + \
                 sorted(img_dir_p.glob("*_0000.nii"))
        lbl_dir_exists = Path(lbl_dir).exists()
        if not lbl_dir_exists:
            print(f"  WARNING: label directory not found: {lbl_dir} -- inference only (no metrics)")
        for img_path in images:
            cid   = case_id_from_image(str(img_path))
            label = find_label(cid, lbl_dir) if lbl_dir_exists else None
            if not label:
                print(f"  NOTE: no fine label found for {cid} in {lbl_dir} -- inference only")
            cases.append((cid, str(img_path), label, subset_tag))

    return cases


# ============================================================
# COLLECT CASES  --  data-dictionary CSV mode
# ============================================================
ALL_CSV_SUBSETS = {"train", "test", "validation", "unlabeled"}


def collect_cases_from_csv(csv_path, subset_filter="all"):
    df = pd.read_csv(csv_path)
    df["subset"] = df["subset"].astype(str).str.strip().str.lower()
    df["subset"] = df["subset"].replace("trains", "train")

    wanted = ALL_CSV_SUBSETS if subset_filter == "all" else \
             {s.strip().lower() for s in subset_filter.split(",")}

    df_filtered = df[df["subset"].isin(wanted)].copy()

    if df_filtered.empty:
        raise ValueError(
            f"No rows found for subset(s) '{subset_filter}' in {csv_path}.\n"
            f"Available values: {sorted(df['subset'].unique())}"
        )

    cases = []
    skipped = 0
    for _, row in df_filtered.iterrows():
        cid      = str(row["ID"])
        vol_path = str(row["vol_path"])
        seg_path = str(row["seg_path"])
        subset   = str(row["subset"])

        if not Path(vol_path).exists():
            print(f"  WARNING: vol not found, skipping {cid}: {vol_path}")
            skipped += 1
            continue

        lbl_path = None if is_placeholder_seg(seg_path) else seg_path
        if lbl_path and not Path(lbl_path).exists():
            print(f"  WARNING: seg not found for {cid}, treating as unlabeled: {seg_path}")
            lbl_path = None

        cases.append((cid, vol_path, lbl_path, subset))

    print(f"  CSV mode: {len(cases)} case(s) loaded from '{csv_path}' "
          f"(subset_filter='{subset_filter}', {skipped} skipped due to missing files)")

    return cases


# ============================================================
# nnUNet RUNNER
# ============================================================

def run_nnunet_predict(input_dir, output_dir, dataset_id):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-d", str(dataset_id),
        "-i", input_dir,
        "-o", output_dir,
        "-f", *[str(f) for f in FOLDS],
        "-c", CONFIGURATION,
        "-tr", TRAINER,
        "-p", PLANS,
        "--continue_prediction",
        "-npp", "1",
        "-nps", "1",
    ]
    print(f"  CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ============================================================
# METRICS
# ============================================================
def vox_vol_mm3(img):
    return float(np.prod(img.header.get_zooms()[:3]))


def surface_distances(pred_bin, gt_bin, spacing):
    if not HAS_SITK:
        return np.nan, np.nan, np.nan
    pred_s = sitk.GetImageFromArray(pred_bin.astype(np.uint8))
    gt_s   = sitk.GetImageFromArray(gt_bin.astype(np.uint8))
    sp_xyz = tuple(reversed(spacing))
    pred_s.SetSpacing(sp_xyz)
    gt_s.SetSpacing(sp_xyz)
    try:
        hdf = sitk.HausdorffDistanceImageFilter()
        hdf.Execute(pred_s, gt_s)
        hd = float(hdf.GetHausdorffDistance())
    except Exception:
        hd = np.nan
    try:
        ps = sitk.LabelContour(pred_s)
        gs = sitk.LabelContour(gt_s)
        dp = sitk.GetArrayFromImage(
            sitk.SignedMaurerDistanceMap(gt_s,   squaredDistance=False, useImageSpacing=True))
        dg = sitk.GetArrayFromImage(
            sitk.SignedMaurerDistanceMap(pred_s, squaredDistance=False, useImageSpacing=True))
        pa = sitk.GetArrayFromImage(ps).astype(bool)
        ga = sitk.GetArrayFromImage(gs).astype(bool)
        d1 = np.abs(dp[pa])
        d2 = np.abs(dg[ga])
        if len(d1) == 0 or len(d2) == 0:
            return np.nan, hd, np.nan
        all_d = np.concatenate([d1, d2])
        return float(np.mean(all_d)), hd, float(np.percentile(all_d, 95))
    except Exception:
        return np.nan, hd, np.nan


def metrics_per_class(pred_data, gt_data, vol_img):
    vv      = vox_vol_mm3(vol_img)
    spacing = tuple(float(z) for z in vol_img.header.get_zooms()[:3])
    out     = {}
    for c in range(1, 17):
        pb = (pred_data == c).astype(np.uint8)
        gb = (gt_data   == c).astype(np.uint8)
        tp = float(np.sum(pb & gb))
        fp = float(np.sum(pb & ~gb.astype(bool)))
        fn = float(np.sum(~pb.astype(bool) & gb))
        ps, gs = int(np.sum(pb)), int(np.sum(gb))

        den  = 2*tp + fp + fn
        dsc  = (2*tp / den)  if den > 0        else np.nan
        prec = (tp / (tp+fp)) if (tp+fp) > 0   else np.nan
        rec  = (tp / (tp+fn)) if (tp+fn) > 0   else np.nan
        vp, vg = ps*vv, gs*vv

        if gs > 0 and ps > 0:
            com_err = float(np.linalg.norm(
                (np.array(center_of_mass(pb)) - np.array(center_of_mass(gb))) * np.array(spacing)))
        elif gs == 0 and ps == 0:
            com_err = 0.0
        else:
            com_err = np.nan

        if gs == 0 and ps == 0:
            asd = hd = hd95 = np.nan
        elif gs == 0 or ps == 0:
            asd = hd = hd95 = np.nan
        else:
            asd, hd, hd95 = surface_distances(pb, gb, spacing)

        out[c] = dict(
            label=FINE_LABEL_NAMES.get(c, str(c)),
            dsc=dsc, asd_mm=asd, hd95_mm=hd95, hd_mm=hd,
            precision=prec, recall=rec,
            vol_pred_mm3=vp, vol_gt_mm3=vg,
            vol_abserr_mm3=abs(vp-vg), com_err_mm=com_err)
    return out


def summarise(per_class):
    def nm(vals):
        v = [x for x in vals if x is not None and not np.isnan(x)]
        return float(np.mean(v)) if v else np.nan
    keys = ["dsc","asd_mm","hd95_mm","hd_mm","precision","recall","vol_abserr_mm3","com_err_mm"]
    s = {}
    for grp, cls in [("all", range(1,17)), ("drg", DRG_CLASSES), ("plexus", PLEXUS_CLASSES)]:
        for k in keys:
            s[f"{grp}_{k}"] = nm([per_class[c][k] for c in cls if c in per_class])
    return s


# ============================================================
# MAIN CASCADE
# ============================================================
def run_cascade(cases, output_dir):
    set_nnunet_env()
    os.makedirs(output_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []

    final_dir  = os.path.join(output_dir, "final_segmentations"); os.makedirs(final_dir, exist_ok=True)
    coarse_dir = os.path.join(output_dir, "stage1_coarse");       os.makedirs(coarse_dir, exist_ok=True)

    labeled   = sum(1 for *_, lbl, _ in cases if lbl)
    unlabeled = len(cases) - labeled
    print(f"\n{'='*65}")
    print(f"  CASCADE INFERENCE  --  {len(cases)} case(s)")
    print(f"  With fine GT labels (will evaluate): {labeled}")
    print(f"  Without GT (inference only):         {unlabeled}")
    print(f"  GT labels source: Dataset202/labelsTr + labelsTs")
    print(f"  Folds used: {FOLDS}")
    print(f"{'='*65}")

    # Persistent work dir (was tempfile.TemporaryDirectory, which was deleted
    # when SLURM killed the job on timeout -- losing all completed predictions).
    # Combined with --continue_prediction this makes inference fully resumable.
    tmp = os.path.join(output_dir, "_work")
    os.makedirs(tmp, exist_ok=True)
    print(f"  Work dir (persistent, resumable): {tmp}")
    with contextlib.nullcontext():
        s1_in  = os.path.join(tmp, "s1_in");  os.makedirs(s1_in,  exist_ok=True)
        s1_out = os.path.join(tmp, "s1_out"); os.makedirs(s1_out, exist_ok=True)
        s2_in  = os.path.join(tmp, "s2_in");  os.makedirs(s2_in,  exist_ok=True)
        s2_out = os.path.join(tmp, "s2_out"); os.makedirs(s2_out, exist_ok=True)

        print("\n[PREP] Staging images (checking gzip)...")
        valid_cases = []
        for cid, img_path, lbl_path, subset in cases:
            dst = os.path.join(s1_in, f"{cid}_0000.nii.gz")
            try:
                safe_save_as_nii_gz(img_path, dst)
                valid_cases.append((cid, img_path, lbl_path, subset))
            except Exception as e:
                print(f"  ERROR loading {cid}: {e} -- skipping.")

        print(f"\n[STAGE 1] Coarse segmentation  (Dataset 201, folds {FOLDS})...")
        run_nnunet_predict(s1_in, s1_out, dataset_id=201)

        print("\n[STAGE 2 PREP] Building 2-channel inputs...")
        stage2_cases = []
        for cid, img_path, lbl_path, subset in valid_cases:
            coarse_path = os.path.join(s1_out, f"{cid}.nii.gz")
            if not os.path.exists(coarse_path):
                print(f"  WARNING: Stage-1 output missing for {cid} -- skipping.")
                continue
            ch0_path = os.path.join(s2_in, f"{cid}_0000.nii.gz")
            safe_save_as_nii_gz(img_path, ch0_path)
            # Load channel-0 back to get its EXACT on-disk header/affine.
            # Forcing channel-1 to inherit these (rather than pred1's own
            # header) avoids the float-precision pixdim drift that nnUNet's
            # exact spacing comparison rejects (e.g. 0.4 vs 0.39999586).
            ch0_img = nib.load(ch0_path)
            pred1 = nib.load(coarse_path)
            coarse_labels = np.unique(pred1.get_fdata().astype(int)).tolist()
            ch1_hdr = ch0_img.header.copy()
            ch1_hdr.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(pred1.get_fdata().astype(np.float32),
                                      ch0_img.affine, ch1_hdr),
                     os.path.join(s2_in, f"{cid}_0001.nii.gz"))
            print(f"  [{subset:>12}] {cid}  coarse labels={coarse_labels}")
            stage2_cases.append((cid, img_path, lbl_path, subset))

        print(f"\n[STAGE 2] Fine 16-class segmentation  (Dataset 202, folds {FOLDS})...")
        run_nnunet_predict(s2_in, s2_out, dataset_id=202)

        print(f"\n[SAVE + EVAL]")
        for cid, img_path, lbl_path, subset in stage2_cases:
            fine_src   = os.path.join(s2_out,    f"{cid}.nii.gz")
            fine_dst   = os.path.join(final_dir,  f"{cid}_seg_fine.nii.gz")
            coarse_src = os.path.join(s1_out,     f"{cid}.nii.gz")
            coarse_dst = os.path.join(coarse_dir, f"{cid}_seg_coarse.nii.gz")

            if not os.path.exists(fine_src):
                print(f"  [{subset:>12}] {cid} -- WARNING: Stage-2 output not found, skipping.")
                continue
            shutil.copy2(fine_src, fine_dst)
            if os.path.exists(coarse_src):
                shutil.copy2(coarse_src, coarse_dst)

            pred_img    = nib.load(fine_dst)
            pred_data   = np.round(pred_img.get_fdata()).astype(int)
            pred_labels = np.unique(pred_data).tolist()

            if lbl_path:
                try:
                    gt_img  = nib.load(lbl_path)
                    gt_data = np.round(gt_img.get_fdata()).astype(int)
                    gt_labels = np.unique(gt_data).tolist()
                    gt_data[gt_data > 16] = 0

                    vol_img = nib.load(img_path)
                    pc  = metrics_per_class(pred_data, gt_data, vol_img)
                    sm  = summarise(pc)

                    for c, m in pc.items():
                        all_rows.append({"subject": cid, "subset": subset, "class_id": c, **m})

                    print(f"  [{subset:>12}] {cid}")
                    print(f"               pred labels={pred_labels}  gt labels={gt_labels}")
                    print(f"               DSC   all:{sm['all_dsc']:.3f}  DRG:{sm['drg_dsc']:.3f}  BP:{sm['plexus_dsc']:.3f}")
                    print(f"               ASD   all:{sm['all_asd_mm']:.2f}mm  DRG:{sm['drg_asd_mm']:.2f}mm  BP:{sm['plexus_asd_mm']:.2f}mm")
                    print(f"               HD95  all:{sm['all_hd95_mm']:.2f}mm  DRG:{sm['drg_hd95_mm']:.2f}mm  BP:{sm['plexus_hd95_mm']:.2f}mm")
                except Exception as e:
                    print(f"  [{subset:>12}] {cid} -- WARNING: could not load GT ({e}), skipping metrics.")
                    print(f"  [{subset:>12}] {cid}  pred labels={pred_labels}  (no GT -- inference only)")
            else:
                print(f"  [{subset:>12}] {cid}  pred labels={pred_labels}  (no GT -- inference only)")

    if all_rows:
        df = pd.DataFrame(all_rows)
        perclass_csv = os.path.join(output_dir, f"metrics_perclass_{ts}.csv")
        df.to_csv(perclass_csv, index=False)

        mcols = ["dsc","asd_mm","hd95_mm","hd_mm","precision","recall",
                 "vol_pred_mm3","vol_gt_mm3","vol_abserr_mm3","com_err_mm"]

        summary_rows = []
        for c in range(1, 17):
            sdf = df[df["class_id"] == c]
            row = {"class_id": c, "label": FINE_LABEL_NAMES.get(c)}
            for m in mcols:
                row[f"{m}_mean"] = sdf[m].mean()
                row[f"{m}_std"]  = sdf[m].std()
            summary_rows.append(row)
        summary_df  = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(output_dir, f"metrics_summary_{ts}.csv")
        summary_df.to_csv(summary_csv, index=False)

        sub_rows = []
        for sname in df["subset"].unique():
            sdf = df[df["subset"] == sname]
            for c in range(1, 17):
                csdf = sdf[sdf["class_id"] == c]
                row  = {"subset": sname, "class_id": c, "label": FINE_LABEL_NAMES.get(c)}
                for m in mcols:
                    row[f"{m}_mean"] = csdf[m].mean()
                    row[f"{m}_std"]  = csdf[m].std()
                sub_rows.append(row)
        sub_df  = pd.DataFrame(sub_rows)
        sub_csv = os.path.join(output_dir, f"metrics_by_subset_{ts}.csv")
        sub_df.to_csv(sub_csv, index=False)

        print(f"\n{'='*72}")
        print(f"  FINAL SUMMARY  ({df['subject'].nunique()} evaluated subjects)")
        print(f"{'='*72}")
        print(f"  {'Structure':<24} {'DSC':>6} {'ASD':>7} {'HD95':>7} {'CoM':>7} {'VolErr':>10}")
        print(f"  {'-'*62}")
        for _, row in summary_df.iterrows():
            print(f"  {row['label']:<24} {row['dsc_mean']:>6.3f} {row['asd_mm_mean']:>7.2f} "
                  f"{row['hd95_mm_mean']:>7.2f} {row['com_err_mm_mean']:>7.2f} "
                  f"{row['vol_abserr_mm3_mean']:>10.1f}")

        drg_df = df[df["class_id"].isin(DRG_CLASSES)]
        bp_df  = df[df["class_id"].isin(PLEXUS_CLASSES)]
        print(f"\n  {'GROUP':<24} {'DSC':>6} {'ASD':>7} {'HD95':>7} {'CoM':>7} {'VolErr':>10}")
        print(f"  {'-'*62}")
        for name, gdf in [("DRG (all)", drg_df), ("Brachial Plexus", bp_df), ("Overall", df)]:
            print(f"  {name:<24} {gdf['dsc'].mean():>6.3f} {gdf['asd_mm'].mean():>7.2f} "
                  f"{gdf['hd95_mm'].mean():>7.2f} {gdf['com_err_mm'].mean():>7.2f} "
                  f"{gdf['vol_abserr_mm3'].mean():>10.1f}")

        print(f"\n  Per-class CSV:   {perclass_csv}")
        print(f"  Summary CSV:     {summary_csv}")
        print(f"  Per-subset CSV:  {sub_csv}")
    else:
        print("\n  NOTE: No GT labels found for any processed case -- no metrics CSVs written.")

    print(f"\n  Final segs:   {final_dir}")
    print(f"  Coarse segs:  {coarse_dir}\n")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cascaded nnUNet inference: DRG + Brachial Plexus")
    parser.add_argument("--output",   required=True,
                        help="Root output directory")
    parser.add_argument("--subset",   default="all",
                        help=(
                            "Which split(s) to run.\n"
                            "  Without --datadict : all | train | test\n"
                            "  With    --datadict : all | train | test | validation | unlabeled\n"
                            "                       or comma-separated, e.g. 'train,validation'\n"
                            "(default: all)"))
    parser.add_argument("--datadict", default=None,
                        help=(
                            "Path to data-dictionary CSV "
                            "(columns: ID, vol_path, seg_path, subset).\n"
                            "When provided, cases are sourced from this CSV instead of "
                            "the nnUNet raw directories.\n"
                            "Rows whose seg_path is the placeholder dummy are treated as "
                            "unlabeled (inference only, no metrics)."))
    parser.add_argument("--input",    default=None,
                        help="Single .nii.gz or folder (bypasses nnUNet raw dirs and --datadict)")
    parser.add_argument("--gt",       default=None,
                        help="Folder of fine GT labels when using --input")
    args = parser.parse_args()

    set_nnunet_env()

    if args.input:
        p    = Path(args.input)
        imgs = [str(p)] if p.is_file() else \
               [str(x) for x in sorted(p.glob("*.nii.gz")) + sorted(p.glob("*.nii"))]
        cases = []
        for img in imgs:
            cid = case_id_from_image(img)
            lbl = find_label(cid, args.gt) if args.gt else None
            cases.append((cid, img, lbl, "manual"))

    elif args.datadict:
        print(f"\n[INFO] Data-dictionary mode: {args.datadict}")
        print(f"[INFO] Subset filter: '{args.subset}'")
        cases = collect_cases_from_csv(args.datadict, subset_filter=args.subset)

    else:
        if args.subset not in ("all", "train", "test"):
            parser.error(
                f"--subset '{args.subset}' requires --datadict to be specified.\n"
                f"Without --datadict only 'all', 'train', or 'test' are supported."
            )
        cases = collect_cases(subset=args.subset)

    if not cases:
        print("ERROR: No cases found. Check directory paths / CSV content.")
        raise SystemExit(1)

    print(f"Found {len(cases)} case(s)  (subset='{args.subset}')")
    run_cascade(cases, args.output)
