#!/usr/bin/env python3
"""
For each subject, uses the manual/expert segmentation (nnU-Net labelsTr/labelsTs)
if one exists -- that's what the model was trained/evaluated against, so it's
the highest-quality input for the normative model. Falls back to the automated
cascade prediction (_seg_fine) only for subjects with no manual label (the
expansion cohort segmented purely by inference).

Sources, in priority order:
  1. manual ground truth : <gt-dir>/<SUBJECT>.nii.gz          (labelsTr, labelsTs)
  2. automated prediction: <seg16-dir>/<SUBJECT>_seg_fine.nii.gz (fallback)

Other inputs unchanged:
  - native STIR : <images-dir>/<SUBJECT>_0000.nii.gz
  - vertebrae   : <tss-dir>/<SUBJECT>/tss_output/step2_output/*.nii.gz (1mm ISO)

manifest.csv gets a `seg_source` column ("manual_gt" or "nnunet_pred") so you
can report the split and, for subjects with BOTH, later compute prediction
quality (Dice) against the manual label if useful.

Run:
    python 00_inventory.py \
      --tss-dir  /scratch/users/jfundaun/bpseg/derivatives/totalspineseg \
      --images-dir /scratch/.../Dataset202_DRGPlexusFine/imagesTr \
                   /scratch/.../Dataset202_DRGPlexusFine/imagesTs \
      --gt-dirs    /scratch/.../Dataset202_DRGPlexusFine/labelsTr \
                   /scratch/.../Dataset202_DRGPlexusFine/labelsTs \
      --seg16-dir  /scratch/.../cascade_results_final/1July2026/final_segmentations \
      --out manifest.csv
"""
import argparse
import glob
import os
import pandas as pd

try:
    import nibabel as nib
    def shape_of(p):
        try:
            return tuple(int(x) for x in nib.load(p).shape[:3])
        except Exception:
            return None
except Exception:
    def shape_of(p):
        return None


def first(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else ""


def find_vertebrae(tss_dir, subj):
    p = first(os.path.join(tss_dir, subj, "tss_output", "step2_output", "*.nii.gz"))
    if not p:
        p = first(os.path.join(tss_dir, subj, "**", "step2_output", "*.nii.gz"))
    return p


def find_stir(images_dirs, subj):
    for d in images_dirs:
        p = first(os.path.join(d, f"{subj}_0000.nii.gz"))
        if p:
            return p
    return ""


def find_seg(gt_dirs, seg16_dir, subj):
    """Return (path, source). Manual ground truth wins if present."""
    for d in gt_dirs:
        p = os.path.join(d, f"{subj}.nii.gz")
        if os.path.exists(p):
            return p, "manual_gt"
    if seg16_dir:
        p = os.path.join(seg16_dir, f"{subj}_seg_fine.nii.gz")
        if os.path.exists(p):
            return p, "nnunet_pred"
    return "", ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tss-dir", required=True)
    ap.add_argument("--images-dir", nargs="+", required=True)
    ap.add_argument("--gt-dirs", nargs="*", default=[],
                    help="manual/expert label dirs (labelsTr labelsTs), highest priority")
    ap.add_argument("--seg16-dir", default="",
                    help="automated cascade prediction dir, used only if no GT label")
    ap.add_argument("--metadata", default="")
    ap.add_argument("--out", default="manifest.csv")
    args = ap.parse_args()

    subjects = sorted(d for d in os.listdir(args.tss_dir)
                      if os.path.isdir(os.path.join(args.tss_dir, d)))
    meta = pd.read_csv(args.metadata).set_index("subject_id") if args.metadata else None

    rows = []
    for subj in subjects:
        stir = find_stir(args.images_dir, subj)
        vert = find_vertebrae(args.tss_dir, subj)
        seg16, source = find_seg(args.gt_dirs, args.seg16_dir, subj)
        ss = shape_of(stir) if stir else None
        gs = shape_of(seg16) if seg16 else None
        native_match = (ss == gs) if (ss and gs) else None

        row = {"subject_id": subj, "stir_path": stir, "vert_path": vert,
               "seg16_path": seg16, "seg_source": source,
               "stir_shape": ss, "seg16_shape": gs,
               "native_stir_seg16_match": native_match,
               "files_present": bool(stir and vert and seg16)}
        if meta is not None and subj in meta.index:
            for c in ("age", "sex", "scanner", "acquisition", "is_hc"):
                if c in meta.columns:
                    row[c] = meta.loc[subj, c]
        meta_ok = (meta is not None and all(
            (row.get(c) == row.get(c)) and row.get(c) not in ("", None)
            for c in ("age", "sex", "scanner", "acquisition")))
        row["metadata_ok"] = bool(meta_ok)
        row["ready"] = bool(row["files_present"] and native_match in (True, None))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    df.loc[df.ready, "subject_id"].to_csv("runnable_subjects.txt", index=False, header=False)

    n = len(df)
    print(f"subjects (tss dirs)         : {n}")
    print(f"  native STIR present       : {int(df.stir_path.astype(bool).sum())}")
    print(f"  vertebrae present         : {int(df.vert_path.astype(bool).sum())}")
    print(f"  seg: manual ground truth  : {int((df.seg_source=='manual_gt').sum())}")
    print(f"  seg: nnU-Net prediction   : {int((df.seg_source=='nnunet_pred').sum())}")
    print(f"  seg: MISSING (neither)    : {int((df.seg_source=='').sum())}")
    if df.native_stir_seg16_match.notna().any():
        print(f"  native STIR/seg MISMATCH  : {int((df.native_stir_seg16_match == False).sum())}")
    print(f"  metadata complete         : {int(df.metadata_ok.sum())}")
    print(f"READY to extract            : {int(df.ready.sum())}  -> runnable_subjects.txt")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
