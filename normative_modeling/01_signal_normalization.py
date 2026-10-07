#!/usr/bin/env python3
"""
Within-subject signal normalization.

Input : STIR image + TotalSpineSeg segmentation (step2_output).
Output: one-row CSV with the subject's reference intensity (+ per-body means for QC). Script 02 reads this file.

Run   : python 01_signal_normalization.py \
            --image sub-01_STIR.nii.gz \
            --vertebrae sub-01_step2_output.nii.gz \
            --subject-id sub-01 \
            --out sub-01_normref.csv

Deps  : SimpleITK, numpy, pandas   (pip install SimpleITK numpy pandas)
"""
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk

# TotalSpineSeg label integers (from tss_map.json): C5=15, C6=16, C7=17, T1=21.
# C5-T1 spans the 4 vertebral bodies flanking the C5-C8 nerve roots.
VERT_REF_LABELS = {"C5": 15, "C6": 16, "C7": 17, "T1": 21}


def compute_reference(image_path, vert_path, ref_labels=VERT_REF_LABELS):
    """Return (pooled_reference_intensity, {body: mean_intensity}).

    Pooled reference = mean STIR over the UNION of C5-T1 body voxels. Pooling
    voxels (rather than averaging four body-means) weights each body by its
    volume and is robust to a body being partly out of FOV.
    """
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float64)
    vert = sitk.GetArrayFromImage(sitk.ReadImage(str(vert_path)))
    if img.shape != vert.shape:
        raise ValueError(
            f"Image {img.shape} and vertebrae mask {vert.shape} are not on the "
            "same grid. Resample the TotalSpineSeg output into the STIR space "
            "first (do NOT use --iso output here; use native-space output)."
        )
    per_body = {}
    for name, lbl in ref_labels.items():
        m = vert == lbl
        per_body[name] = float(img[m].mean()) if m.any() else np.nan

    union = np.isin(vert, list(ref_labels.values()))
    n_vox = int(union.sum())
    if n_vox == 0:
        raise ValueError("No C5-T1 vertebral voxels found -- check label map / FOV.")
    reference = float(img[union].mean())
    return reference, per_body, n_vox


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="STIR NIfTI (native space)")
    ap.add_argument("--vertebrae", required=True,
                    help="TotalSpineSeg step2_output NIfTI (native space)")
    ap.add_argument("--subject-id", required=True)
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    ref, per_body, n_vox = compute_reference(args.image, args.vertebrae)

    row = {"subject_id": args.subject_id,
           "vert_ref_intensity": ref,
           "n_ref_voxels": n_vox}
    row.update({f"ref_{k}": v for k, v in per_body.items()})
    pd.DataFrame([row]).to_csv(args.out, index=False)
    print(f"[{args.subject_id}] C5-T1 reference = {ref:.2f} "
          f"(n={n_vox} voxels) -> {args.out}")


if __name__ == "__main__":
    main()
