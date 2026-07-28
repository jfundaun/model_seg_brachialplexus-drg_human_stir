#!/usr/bin/env python3
"""
Per-structure image-derived phenotypes for the 16 structures (8 DRGs + 8 roots,
bilateral C5-C8).

Each structure's pyRadiomics call (Mean, GLCM JointEntropy) runs in an ISOLATED
SUBPROCESS via extract_one_label.py. This is required because pyRadiomics can
segfault natively on certain masks -- a native crash cannot be caught by a
Python try/except, and would otherwise kill this entire script (losing every
remaining structure for that subject). Running it as a subprocess means any
failure, exception OR crash, becomes just a return code the parent can log and
skip past.

What is computed here (the directly-measurable IDPs):
    SI_norm            #1  all 16   firstorder Mean / C5-T1 reference   [2D+3D]
    SI_entropy         #3  all 16   GLCM JointEntropy within mask       [2D+3D]
    Volume_mm3         (->#8 Volume_z later)  all 16                    [3D]
    Mean_CSA           #4  roots    volume / centerline length          [3D]
    FSI                #6  roots    max(CSA)/median(CSA) along line      [3D]
    Peak_SI_location   #7  roots    normalized position of peak SI       [3D]

Run (one subject):
    python 02_extract_idps.py \
        --image sub-01_STIR.nii.gz \
        --seg sub-01_structures16.nii.gz \
        --normref sub-01_normref.csv \
        --subject-id sub-01 --age 34 --sex F --scanner GE_site1 \
        --acquisition 3D \
        --out sub-01_idps.csv

Requires extract_one_label.py in the same directory.
Deps: SimpleITK, numpy, pandas   (pyradiomics is needed only by the helper)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_one_label.py")

# ---------------------------------------------------------------------------
# Verified against Dataset202_DRGPlexusFine/dataset.json:
#   {'DRG_C5_R':1,'DRG_C6_R':2,'DRG_C7_R':3,'DRG_C8_R':4,
#    'DRG_C5_L':5,'DRG_C6_L':6,'DRG_C7_L':7,'DRG_C8_L':8,
#    'BP_C5_R':9,'BP_C6_R':10,'BP_C7_R':11,'BP_C8_R':12,
#    'BP_C5_L':13,'BP_C6_L':14,'BP_C7_L':15,'BP_C8_L':16}
# BP_* (brachial plexus) -> type="root" (the morphometry structures).
# ---------------------------------------------------------------------------
STRUCTURE_LABELS = {
    1: ("R", "C5", "DRG"),  2: ("R", "C6", "DRG"),
    3: ("R", "C7", "DRG"),  4: ("R", "C8", "DRG"),
    5: ("L", "C5", "DRG"),  6: ("L", "C6", "DRG"),
    7: ("L", "C7", "DRG"),  8: ("L", "C8", "DRG"),
    9:  ("R", "C5", "root"), 10: ("R", "C6", "root"),
    11: ("R", "C7", "root"), 12: ("R", "C8", "root"),
    13: ("L", "C5", "root"), 14: ("L", "C6", "root"),
    15: ("L", "C7", "root"), 16: ("L", "C8", "root"),
}
GLCM_BIN_WIDTH = 5  # STIR intensity bin for GLCM; keep fixed across the cohort.


def extract_one_label_safe(image_path, seg_path, label, subject_id, side, level, stype):
    """Run pyRadiomics for ONE label in a subprocess. Returns (mean, entropy),
    (nan, nan) on any failure (exception OR native crash), with a WARN logged
    to stderr so failures are visible in the SLURM logs instead of silent."""
    try:
        proc = subprocess.run(
            [sys.executable, HELPER, image_path, seg_path, str(label), str(GLCM_BIN_WIDTH)],
            capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] {subject_id} label {label} ({side}_{level}_{stype}) "
              f"TIMEOUT after 120s", file=sys.stderr)
        return np.nan, np.nan

    if proc.returncode != 0:
        tag = "SEGFAULT/CRASH" if proc.returncode < 0 else f"exit {proc.returncode}"
        print(f"  [WARN] {subject_id} label {label} ({side}_{level}_{stype}) "
              f"{tag}: {proc.stderr.strip()[-300:]}", file=sys.stderr)
        return np.nan, np.nan
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        return out["mean"], out["entropy"]
    except Exception as e:
        print(f"  [WARN] {subject_id} label {label} ({side}_{level}_{stype}) "
              f"bad helper output {proc.stdout!r}: {e}", file=sys.stderr)
        return np.nan, np.nan


def centerline_morphometry(mask_arr, img_arr, spacing_zyx, bin_mm=2.0):
    """Root morphometry along the principal (centerline) axis.

    Approximates the nerve-root centerline as the first principal axis of the
    mask voxel cloud (roots are near-tubular), then measures cross-sectional
    area (CSA) perpendicular to it in bins.
        Mean_CSA         = volume / centerline_length            (Table 7 #4)
        FSI              = max(CSA) / median(CSA)                 (Table 7 #6)
        Peak_SI_location = position (0-1) of peak mean SI on line (Table 7 #7)
    Returns dict or None if the mask is too small.
    """
    coords = np.argwhere(mask_arr > 0)  # (N,3) in (z,y,x) index order
    if len(coords) < 20:
        return None
    voxvol = float(np.prod(spacing_zyx))
    phys = coords * np.asarray(spacing_zyx)              # -> mm, (z,y,x)
    X = phys - phys.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    axis = vt[0]
    t = X @ axis                                         # projection (mm)
    length = float(t.max() - t.min())
    if length <= 0:
        return None
    volume = len(coords) * voxvol
    mean_csa = volume / length

    K = max(3, int(round(length / bin_mm)))
    edges = np.linspace(t.min(), t.max(), K + 1)
    b = np.clip(np.digitize(t, edges) - 1, 0, K - 1)
    thick = length / K
    si = img_arr[coords[:, 0], coords[:, 1], coords[:, 2]]
    csa = np.zeros(K)
    mean_si = np.full(K, np.nan)
    for k in range(K):
        m = b == k
        if m.any():
            csa[k] = m.sum() * voxvol / thick            # area perpendicular to axis
            mean_si[k] = si[m].mean()
    valid = csa[csa > 0]
    fsi = float(csa.max() / np.median(valid)) if valid.size else np.nan
    peak_loc = float(np.nanargmax(mean_si) / (K - 1)) if K > 1 else 0.0
    return {"Volume_mm3": volume, "Mean_CSA": mean_csa,
            "FSI": fsi, "Peak_SI_location": peak_loc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--seg", required=True, help="16-structure segmentation NIfTI")
    ap.add_argument("--normref", required=True, help="CSV from script 01")
    ap.add_argument("--subject-id", required=True)
    ap.add_argument("--age", type=float, required=True)
    ap.add_argument("--sex", required=True)
    ap.add_argument("--scanner", required=True)
    ap.add_argument("--acquisition", default="3D", choices=["2D", "3D"],
                    help="morphology IDPs are only valid on 3D (see Section 4.4)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reference = float(pd.read_csv(args.normref)["vert_ref_intensity"].iloc[0])

    sitk_img = sitk.ReadImage(args.image)
    seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(args.seg))
    img_arr = sitk.GetArrayFromImage(sitk_img).astype(np.float64)
    sx, sy, sz = sitk_img.GetSpacing()           # SimpleITK gives (x,y,z)
    spacing_zyx = (sz, sy, sx)                    # array axes are (z,y,x)

    rows = []
    for lbl, (side, level, stype) in STRUCTURE_LABELS.items():
        if not (seg_arr == lbl).any():
            continue

        mean_si, entropy = extract_one_label_safe(
            args.image, args.seg, lbl, args.subject_id, side, level, stype)

        rec = {"subject_id": args.subject_id, "age": args.age, "sex": args.sex,
               "scanner": args.scanner, "acquisition": args.acquisition,
               "structure": f"{side}_{level}_{stype}", "side": side,
               "level": level, "type": stype,
               "SI_norm": (mean_si / reference) if (reference and mean_si == mean_si) else np.nan,
               "SI_entropy": entropy,
               "Volume_mm3": np.nan, "Mean_CSA": np.nan,
               "FSI": np.nan, "Peak_SI_location": np.nan}

        # --- morphometry (3D only; roots use centerline, DRGs use volume) ---
        if args.acquisition == "3D":
            morph = centerline_morphometry(seg_arr == lbl, img_arr, spacing_zyx)
            if morph:
                if stype == "root":
                    rec.update(morph)             # Volume, Mean_CSA, FSI, Peak_SI_location
                else:                             # DRG: volume only (-> Volume_z)
                    rec["Volume_mm3"] = morph["Volume_mm3"]
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    n_nan = int(df["SI_norm"].isna().sum())
    print(f"[{args.subject_id}] extracted {len(df)} structures "
          f"({n_nan} with failed SI_norm) -> {args.out}")


if __name__ == "__main__":
    main()
