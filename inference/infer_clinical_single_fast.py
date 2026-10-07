#!/usr/bin/env python3
"""
infer_clinical_single_fast.py
==============================
Cascaded nnUNet inference (Dataset201 coarse -> Dataset202 fine, 16-class
DRG + Brachial Plexus) for a single clinical NIfTI image.

PRESETS (choose with --preset; individual flags override):
  reference : coarse=0-4, fine=0-4, TTA on,  step 0.5   (reproduces original)
  balanced  : coarse=0,   fine=0-2, TTA off, step 0.7   (default)
  fast      : coarse=0,   fine=0,   TTA off, step 0.7

Usage:
python infer_clinical_single_fast.py --preset balanced \
  --input /path/file.nii.gz \
  --output /path/

  # reproduce the original 5+5 fold + TTA config for comparison:
python infer_clinical_single_fast.py --preset reference --input ... --output ...
"""

import os
import re
import time
import shutil
import tempfile
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label as scipy_label

# ============================================================
# CONFIGURATION
# ============================================================

NNUNET_RESULTS = (
    "/path_here/"
)

TRAINER       = "nnUNetTrainer"
PLANS         = "nnUNetResEncUNetMPlans"
CONFIGURATION = "3d_fullres"

FINE_LABEL_NAMES = {
    0:  "background",
    1:  "DRG_C5_R",  2:  "DRG_C6_R",  3:  "DRG_C7_R",  4:  "DRG_C8_R",
    5:  "DRG_C5_L",  6:  "DRG_C6_L",  7:  "DRG_C7_L",  8:  "DRG_C8_L",
    9:  "BP_C5_R",   10: "BP_C6_R",   11: "BP_C7_R",   12: "BP_C8_R",
    13: "BP_C5_L",   14: "BP_C6_L",   15: "BP_C7_L",   16: "BP_C8_L",
}

# Each tuple: (right_class, left_class) -- same spinal level
BILATERAL_PAIRS = [
    (1, 5), (2, 6), (3, 7), (4, 8),         # DRG  C5-C8
    (9, 13), (10, 14), (11, 15), (12, 16),  # BP   C5-C8
]

PRESETS = {
    "reference": dict(coarse_folds=(0, 1, 2, 3, 4),
                      fine_folds=(0, 1, 2, 3, 4), tta=True,  step=0.5),
    "balanced":  dict(coarse_folds=(0,),
                      fine_folds=(0, 1, 2),       tta=False, step=0.7),
    "fast":      dict(coarse_folds=(0,),
                      fine_folds=(0,),            tta=False, step=0.7),
}


# ============================================================
# MODEL / FOLD HELPERS
# ============================================================

def model_folder(dataset_id):
    ds_name = ("Dataset201_DRGPlexusCoarse" if dataset_id == 201
               else "Dataset202_DRGPlexusFine")
    return os.path.join(
        NNUNET_RESULTS, ds_name, f"{TRAINER}__{PLANS}__{CONFIGURATION}"
    )


def check_folds(dataset_id, folds):
    """Lightweight existence check (no torch.load). Returns the subset of
    requested folds whose checkpoint_final.pth exists."""
    base = model_folder(dataset_id)
    ok = []
    for f in folds:
        ckpt = Path(base) / f"fold_{f}" / "checkpoint_final.pth"
        if ckpt.exists():
            ok.append(f)
            print(f"  Dataset{dataset_id} fold_{f}: found")
        else:
            print(f"  Dataset{dataset_id} fold_{f}: MISSING checkpoint, skipping")
    if not ok:
        raise RuntimeError(
            f"No requested folds available for Dataset{dataset_id} under {base}"
        )
    return tuple(ok)


def get_device(name):
    import torch
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "ERROR: --device cuda requested but torch.cuda.is_available() "
                "is False. Check CUDA_VISIBLE_DEVICES on this node."
            )
        print(f"  Device: cuda ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    print("  Device: cpu (WARNING: 3d_fullres on CPU is very slow)")
    return torch.device("cpu")


def build_predictor(dataset_id, folds, use_tta, step_size, device):
    """Construct and initialise a predictor ONCE. Reuse across cases."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=use_tta,                       # <-- TTA switch
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder(dataset_id),
        use_folds=folds,                             # <-- fold ensemble
        checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def predict_folder(predictor, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    predictor.predict_from_files(
        str(in_dir),
        str(out_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


# ============================================================
# ENVIRONMENT
# ============================================================

def set_nnunet_env(raw_dir, preproc_dir):
    os.environ["nnUNet_results"]      = NNUNET_RESULTS
    os.environ["nnUNet_raw"]          = str(raw_dir)
    os.environ["nnUNet_preprocessed"] = str(preproc_dir)


# ============================================================
# NIFTI / ORIENTATION HELPERS 
# ============================================================

def stage_image(src, dst):
    """Reorient to RAS+ canonical and save as .nii.gz."""
    img     = nib.load(str(src))
    img_ras = nib.as_closest_canonical(img)
    orig    = nib.aff2axcodes(img.affine)
    new     = nib.aff2axcodes(img_ras.affine)
    if orig != new:
        print(f"    [reorient] {Path(src).name}: {orig} -> {new}")
    nib.save(img_ras, str(dst))
    return img_ras.affine


def save_coarse_channel(coarse_img, dst_path):
    """Save coarse seg as uint8 with scl_slope/scl_inter=0 (prevents
    float label corruption, e.g. 1 -> 1.0000152587890625)."""
    arr = np.round(coarse_img.get_fdata()).astype(np.uint8)
    hdr = coarse_img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr["scl_slope"] = 0.0
    hdr["scl_inter"] = 0.0
    hdr.set_qform(coarse_img.affine, code=1)  # Always set to 1 (Scanner Anat)
    hdr.set_sform(coarse_img.affine, code=1)  # Always set to 1
    nib.save(nib.Nifti1Image(arr, coarse_img.affine, hdr), str(dst_path))


def affines_close(a1, a2, tol=1e-3):
    return np.allclose(a1, a2, atol=tol)


def sanitize_case_id(filename):
    stem = Path(filename).name
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    if stem.endswith("_0000"):
        stem = stem[:-5]
    stem = re.sub(r"[\s()\[\]]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


# ============================================================
# BILATERAL LATERALIZATION
# ============================================================

def lateralize_bilateral_structures(seg_data, affine):
    """
    For each bilateral R/L pair at the same spinal level, find connected
    components of the combined R+L mask and reassign each component to the
    correct side based on world-space x-coordinate (RAS: +x = Right).
    """
    seg_fixed = seg_data.copy()
    shape     = seg_data.shape[:3]

    ii, jj, kk = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing="ij",
    )
    world_x = (affine[0, 0] * ii +
               affine[0, 1] * jj +
               affine[0, 2] * kk +
               affine[0, 3])

    right_half = world_x >= 0
    left_half  = world_x <  0

    n_splits = 0
    n_relabels = 0

    for r_cls, l_cls in BILATERAL_PAIRS:
        combined = (seg_data == r_cls) | (seg_data == l_cls)
        if not combined.any():
            continue

        labeled_arr, n_comps = scipy_label(combined)

        seg_fixed[seg_data == r_cls] = 0
        seg_fixed[seg_data == l_cls] = 0

        for comp_id in range(1, n_comps + 1):
            comp = labeled_arr == comp_id
            r_vox = int(np.sum(comp & right_half))
            l_vox = int(np.sum(comp & left_half))
            total = r_vox + l_vox
            if total == 0:
                continue

            r_frac = r_vox / total

            if r_frac >= 0.80:
                seg_fixed[comp] = r_cls
                n_relabels += 1
            elif r_frac <= 0.20:
                seg_fixed[comp] = l_cls
                n_relabels += 1
            else:
                seg_fixed[comp & right_half] = r_cls
                seg_fixed[comp & left_half]  = l_cls
                n_splits += 1
                print(
                    f"    [lateralize] Split merged blob "
                    f"cls({r_cls},{l_cls}): {r_vox} R-vox / {l_vox} L-vox"
                )

    if n_relabels > 0:
        print(f"    [lateralize] Re-labelled {n_relabels} component(s) by centroid side")
    if n_splits == 0 and n_relabels == 0:
        print(f"    [lateralize] No bilateral merging detected")

    return seg_fixed


# ============================================================
# PER-CASE PROCESSING
# ============================================================

def process_case(input_path, output_path, predictor_coarse, predictor_fine):
    """Run the full cascade for one case using pre-built predictors."""
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"ERROR: input not found: {input_path}")

    cid = sanitize_case_id(input_path.name)

    final_dir  = output_path / "final_segmentations"
    coarse_dir = output_path / "stage1_coarse"
    final_dir.mkdir(parents=True, exist_ok=True)
    coarse_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  SINGLE-CASE CASCADE INFERENCE")
    print(f"  Input   -> {input_path}")
    print(f"  Case ID -> {cid}")
    print(f"  Output  -> {output_path}")
    print(f"{'='*70}")

    with tempfile.TemporaryDirectory(prefix="nnunet_clinical_") as tmp_str:
        tmp = Path(tmp_str)
        s1_in  = tmp / "s1_in";  s1_in.mkdir()
        s1_out = tmp / "s1_out"; s1_out.mkdir()
        s2_in  = tmp / "s2_in";  s2_in.mkdir()
        s2_out = tmp / "s2_out"; s2_out.mkdir()

        # ── Stage 1 input prep: reorient to RAS+ ────────────────────────────
        print("\n[PREP] Reorienting input to RAS+ canonical...")
        dst = s1_in / f"{cid}_0000.nii.gz"
        stage_image(input_path, dst)

        # ── Stage 1: coarse segmentation ────────────────────────────────────
        print("\n[STAGE 1] Coarse segmentation (Dataset201)...")
        t0 = time.perf_counter()
        predict_folder(predictor_coarse, s1_in, s1_out)
        t_s1 = time.perf_counter() - t0
        print(f"  Stage 1 wall time: {t_s1:6.1f}s")

        coarse_path = s1_out / f"{cid}.nii.gz"
        if not coarse_path.exists():
            raise RuntimeError(f"Stage-1 output missing for {cid}: {coarse_path}")

        # ── Stage 2 input prep: T2 + coarse seg ─────────────────────────────
        print("\n[STAGE 2 PREP] Building 2-channel input (T2 + coarse seg)...")
        ch0_path = s2_in / f"{cid}_0000.nii.gz"
        ch1_path = s2_in / f"{cid}_0001.nii.gz"

        shutil.copy2(str(dst), str(ch0_path))

        coarse_img = nib.load(str(coarse_path))
        coarse_labels = np.unique(
            np.round(coarse_img.get_fdata()).astype(int)).tolist()
        save_coarse_channel(coarse_img, ch1_path)

        ch0_img = nib.load(str(ch0_path))
        ch1_img = nib.load(str(ch1_path))
        if not affines_close(ch0_img.affine, ch1_img.affine):
            print(f"  WARNING [{cid}]: Stage-2 channel affines MISALIGNED.")
            print(f"    ch0 affine:\n{ch0_img.affine}")
            print(f"    ch1 affine:\n{ch1_img.affine}")
        else:
            print(f"  Channel affines aligned OK.")
        print(f"  coarse labels = {coarse_labels}")

        # ── Stage 2: fine 16-class segmentation ─────────────────────────────
        print("\n[STAGE 2] Fine 16-class segmentation (Dataset202)...")
        t0 = time.perf_counter()
        predict_folder(predictor_fine, s2_in, s2_out)
        t_s2 = time.perf_counter() - t0
        print(f"  Stage 2 wall time: {t_s2:6.1f}s")

        fine_src = s2_out / f"{cid}.nii.gz"
        if not fine_src.exists():
            raise RuntimeError(f"Stage-2 output missing for {cid}: {fine_src}")

        # ── Lateralize bilateral structures ─────────────────────────────────
        print("\n[LATERALIZE] Checking for merged bilateral structures...")
        pred_img  = nib.load(str(fine_src))
        pred_data = np.round(pred_img.get_fdata(dtype=np.float32)).astype(int)

        labels_before = sorted(np.unique(pred_data).tolist())
        pred_data_fixed = lateralize_bilateral_structures(pred_data, pred_img.affine)
        labels_after = sorted(np.unique(pred_data_fixed).tolist())

        if labels_before != labels_after:
            print(f"    labels: {labels_before} -> {labels_after}")

        # ── Save final outputs ──────────────────────────────────────────────
        fine_dst   = final_dir  / f"{cid}_seg_fine.nii.gz"
        coarse_dst = coarse_dir / f"{cid}_seg_coarse.nii.gz"

        fixed_hdr = pred_img.header.copy()
        fixed_hdr["scl_slope"] = 0.0
        fixed_hdr["scl_inter"] = 0.0
        fixed_hdr.set_data_dtype(np.int16)
        nib.save(
            nib.Nifti1Image(pred_data_fixed.astype(np.int16),
                            pred_img.affine, fixed_hdr),
            str(fine_dst),
        )
        shutil.copy2(str(coarse_path), str(coarse_dst))

        present = {FINE_LABEL_NAMES.get(c, str(c)) for c in labels_after if c != 0}
        print(f"\n  Final labels present: {sorted(present)}")

    print(f"\n{'='*70}")
    print(f"  DONE  (stage1 {t_s1:.1f}s + stage2 {t_s2:.1f}s = {t_s1 + t_s2:.1f}s)")
    print(f"  Fine seg:   {fine_dst}")
    print(f"  Coarse seg: {coarse_dst}")
    print(f"{'='*70}\n")


# ============================================================
# MAIN
# ============================================================

def parse_folds(s):
    if s is None:
        return None
    if s.strip().lower() == "all":
        return (0, 1, 2, 3, 4)
    return tuple(int(x) for x in re.split(r"[,\s]+", s.strip()) if x != "")


def main():
    parser = argparse.ArgumentParser(
        description="Single-case cascaded nnUNet inference (fast): DRG + Brachial Plexus")
    parser.add_argument("--input",  required=True,
                        help="Path to a single .nii.gz clinical image")
    parser.add_argument("--output", required=True,
                        help="Output directory (e.g. derivatives/<STS_code>)")
    parser.add_argument("--preset", default="balanced",
                        choices=list(PRESETS.keys()),
                        help="Speed/accuracy preset (default: balanced)")
    parser.add_argument("--coarse-folds", default=None,
                        help="Override coarse folds, e.g. '0' or '0,1,2' or 'all'")
    parser.add_argument("--fine-folds", default=None,
                        help="Override fine folds, e.g. '0,1,2' or 'all'")
    parser.add_argument("--tta", dest="tta", action="store_true", default=None,
                        help="Force TTA (mirroring) ON")
    parser.add_argument("--no-tta", dest="tta", action="store_false", default=None,
                        help="Force TTA OFF")
    parser.add_argument("--step-size", type=float, default=None,
                        help="Override tile_step_size (default from preset)")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu", "mps"])
    args = parser.parse_args()

    # preset defaults, then apply any explicit overrides
    cfg = dict(PRESETS[args.preset])
    if args.coarse_folds is not None:
        cfg["coarse_folds"] = parse_folds(args.coarse_folds)
    if args.fine_folds is not None:
        cfg["fine_folds"] = parse_folds(args.fine_folds)
    if args.tta is not None:
        cfg["tta"] = args.tta
    if args.step_size is not None:
        cfg["step"] = args.step_size

    print(f"\n[CONFIG] preset={args.preset}")
    print(f"  coarse_folds = {cfg['coarse_folds']}")
    print(f"  fine_folds   = {cfg['fine_folds']}")
    print(f"  TTA          = {cfg['tta']}")
    print(f"  step_size    = {cfg['step']}")

    device = get_device(args.device)

    # temp env dirs
    with tempfile.TemporaryDirectory(prefix="nnunet_env_") as env_str:
        env = Path(env_str)
        raw_dir     = env / "raw";     raw_dir.mkdir()
        preproc_dir = env / "preproc"; preproc_dir.mkdir()
        set_nnunet_env(raw_dir, preproc_dir)

        print("\n[FOLD CHECK]")
        folds_201 = check_folds(201, cfg["coarse_folds"])
        folds_202 = check_folds(202, cfg["fine_folds"])

        # build BOTH predictors ONCE (this is where the speedup lives)
        print("\n[BUILD] Initialising coarse predictor (Dataset201)...")
        t0 = time.perf_counter()
        predictor_coarse = build_predictor(201, folds_201, cfg["tta"],
                                            cfg["step"], device)
        print(f"  coarse predictor ready ({time.perf_counter() - t0:.1f}s)")

        print("[BUILD] Initialising fine predictor (Dataset202)...")
        t0 = time.perf_counter()
        predictor_fine = build_predictor(202, folds_202, cfg["tta"],
                                          cfg["step"], device)
        print(f"  fine predictor ready ({time.perf_counter() - t0:.1f}s)")

        # ---- single case ----
        process_case(args.input, args.output, predictor_coarse, predictor_fine)

        # ---- cohort loop (uncomment; predictors are already built once) ----
        # cases = [
        #     ("/.../nifti/CASE1/img.nii.gz", "/.../derivatives/CASE1"),
        #     ("/.../nifti/CASE2/img.nii.gz", "/.../derivatives/CASE2"),
        # ]
        # for in_p, out_p in cases:
        #     process_case(in_p, out_p, predictor_coarse, predictor_fine)


if __name__ == "__main__":
    main()
