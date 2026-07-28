#!/usr/bin/env python3
"""
Replace Dataset202 channel 1 (_0001) files with Stage 1 cross-val predictions.
=================================================================================
Run this AFTER cascade_pipeline.sh Step 1 has generated merged Stage 1 preds.
Run BEFORE nnUNetv2_preprocess -d 202.

ALL FIXES vs previous versions:
  1. Exact name matching before fuzzy glob — prevents wrong-subject assignment
  2. Float corruption fix: scl_slope/scl_inter zeroed on all saved files
  3. RAS+ reorientation of Stage 1 pred before resampling to MRI space
  4. Backup of original channel 1 files before overwrite
  5. Per-file integrity verification after save
  6. Summary flags cases that kept GT coarse (train/inference mismatch risk)
"""

import os
import sys
import shutil
import glob
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates
from datetime import datetime

# ============================================================
# PATHS — edit to match your environment
# ============================================================
BASE     = "/scratch/users/jfundaun/bpseg/nnunet_cascade/raw/Dataset202_DRGPlexusFine"
PRED_DIR = "/scratch/users/jfundaun/bpseg/derivatives/stage1_crossval_preds/merged"
BACKUP_DIR = "/scratch/users/jfundaun/bpseg/derivatives/channel1_backup_gt_coarse"
# ============================================================


def reorient_to_ras(img):
    """Reorient to RAS+ canonical orientation."""
    return nib.as_closest_canonical(img)


def resample_to_ref(src_img, ref_img):
    """
    Nearest-neighbour resample src into ref_img voxel space.
    Both inputs must already be in RAS+ orientation before calling.
    Returns a NiftiImage in ref_img space.
    """
    if (np.allclose(src_img.affine, ref_img.affine, atol=1e-3) and
            src_img.shape[:3] == ref_img.shape[:3]):
        # Force ref header/affine so saved _0001 has EXACT _0000 geometry
        # (avoids ~1e-6 zoom mismatch that nnUNet rejects)
        return nib.Nifti1Image(
            np.round(src_img.get_fdata()).astype(np.uint8),
            ref_img.affine, ref_img.header)

    ref_shape  = ref_img.shape[:3]
    label_data = np.round(src_img.get_fdata()).astype(np.int32)

    i, j, k = np.meshgrid(
        np.arange(ref_shape[0]),
        np.arange(ref_shape[1]),
        np.arange(ref_shape[2]),
        indexing="ij"
    )
    ref_vox_hom = np.stack(
        [i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)], axis=0)
    world_coords = ref_img.affine @ ref_vox_hom
    src_vox      = (np.linalg.inv(src_img.affine) @ world_coords)[:3, :]

    resampled = map_coordinates(
        label_data, src_vox, order=0, mode="constant", cval=0
    ).reshape(ref_shape).astype(np.uint8)

    return nib.Nifti1Image(resampled, ref_img.affine, ref_img.header)


def save_clean_seg(data, ref_img, path):
    """
    Save segmentation with zeroed scl_slope/scl_inter.
    Prevents float corruption on reload (1 → 1.0000152587890625 bug).
    """
    arr = data.astype(np.uint8)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr["scl_slope"] = 0.0   # 0 = "no scaling" per NIfTI-1 spec
    hdr["scl_inter"] = 0.0
    hdr.set_qform(ref_img.affine, code=max(1, int(hdr["qform_code"])))
    hdr.set_sform(ref_img.affine, code=max(1, int(hdr["sform_code"])))
    # Force pixdim to EXACTLY match reference zooms. nnUNet reads spacing
    # from pixdim (get_zooms), not the affine, and reconstructing an image
    # recomputes pixdim from affine column norms -> tiny float drift
    # (0.4 -> 0.39999586) that nnUNet rejects as a spacing mismatch.
    out = nib.Nifti1Image(arr, ref_img.affine, hdr)
    ref_zooms = ref_img.header.get_zooms()[:3]
    z = list(out.header.get_zooms())
    z[:3] = ref_zooms
    out.header.set_zooms(tuple(z))
    nib.save(out, path)


def verify_clean(path):
    """Return (is_clean, max_rounding_error, max_label)."""
    d   = nib.load(path).get_fdata(dtype=np.float32)
    err = float(np.abs(d - np.round(d)).max())
    mx  = int(d.max()) if d.size > 0 else 0
    return err < 1e-4 and mx <= 2, err, mx


def find_pred_exact_then_fuzzy(sub_id, pred_dir):
    """
    Find Stage 1 prediction for sub_id.

    Exact patterns tried first (in order):
      1. <sub_id>.nii.gz
      2. <sub_id>_seg_coarse.nii.gz
      3. <sub_id>_seg.nii.gz

    Fuzzy fallback: *<sub_id>*.nii.gz
      - If exactly 1 match: use it, no warning
      - If >1 match: use first alphabetically, WARN (ambiguous)
      - If 0 matches: return None

    This prevents the original bug where sub_id='003' matched
    '003_010_t2.nii.gz', '003_020_t2.nii.gz' etc. arbitrarily.
    """
    pred_dir = Path(pred_dir)

    for name in (f"{sub_id}.nii.gz",
                 f"{sub_id}_seg_coarse.nii.gz",
                 f"{sub_id}_seg.nii.gz"):
        p = pred_dir / name
        if p.exists():
            return str(p), "exact"

    hits = sorted(pred_dir.glob(f"*{sub_id}*.nii.gz"))
    if len(hits) == 1:
        return str(hits[0]), "fuzzy"
    if len(hits) > 1:
        names = [h.name for h in hits]
        return str(hits[0]), f"fuzzy_ambiguous({len(hits)}:{names})"
    return None, "not_found"


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(BACKUP_DIR, exist_ok=True)

    print("=" * 65)
    print("Dataset202 Channel 1 Update — Stage 1 Cross-Val Predictions")
    print(f"  Source: {PRED_DIR}")
    print(f"  Target: {BASE}")
    print(f"  Backup: {BACKUP_DIR}")
    print("=" * 65)

    if not Path(PRED_DIR).exists():
        print(f"ERROR: PRED_DIR does not exist: {PRED_DIR}")
        sys.exit(1)

    pred_files = list(Path(PRED_DIR).glob("*.nii.gz"))
    print(f"\nStage 1 predictions available: {len(pred_files)}")

    total = ok = missing = bad = fuzzy_warn = 0
    kept_gt = []   # subjects that kept GT coarse — train/inference mismatch risk

    for split in ["imagesTr", "imagesTs"]:
        ch1_files = sorted(glob.glob(f"{BASE}/{split}/*_0001.nii.gz"))
        print(f"\n{split}: {len(ch1_files)} channel-1 files to update")

        for ch1_path in ch1_files:
            # Derive subject ID from filename
            name = Path(ch1_path).name           # e.g. 003_010_t2_0001.nii.gz
            subj = name.replace("_0001.nii.gz", "").replace("_0001.nii", "")
            ch0_path = ch1_path.replace("_0001.nii.gz", "_0000.nii.gz")
            total += 1

            # Find Stage 1 prediction
            pred_path, match_type = find_pred_exact_then_fuzzy(subj, PRED_DIR)

            if pred_path is None:
                print(f"  MISSING  {subj}")
                missing += 1
                kept_gt.append(subj)
                continue

            if "ambiguous" in match_type:
                print(f"  WARN ambiguous match: {subj} → {match_type}")
                fuzzy_warn += 1

            # Load prediction and reference MRI
            pred_img = reorient_to_ras(nib.load(pred_path))
            ref_img  = reorient_to_ras(nib.load(ch0_path))  # MRI _0000

            # Resample Stage 1 pred to MRI space (both already RAS+)
            pred_resampled = resample_to_ref(pred_img, ref_img)
            pred_data = np.round(
                pred_resampled.get_fdata(dtype=np.float32)).astype(np.uint8)

            # Validate labels
            labels = sorted(int(v) for v in np.unique(pred_data))
            if max(labels, default=0) > 2:
                print(f"  BAD labels {subj}: {labels} — skipping")
                bad += 1
                kept_gt.append(subj)
                continue

            # Backup original channel 1 before overwriting
            backup_path = os.path.join(BACKUP_DIR, f"{subj}_0001_gt_coarse.nii.gz")
            if not Path(backup_path).exists():
                shutil.copy2(ch1_path, backup_path)

            # Save with clean header
            save_clean_seg(pred_data, ref_img, ch1_path)

            # Verify integrity after save
            is_clean, err, mx = verify_clean(ch1_path)
            if not is_clean:
                print(f"  CORRUPT after save {subj}: err={err:.8f} max_label={mx}")
                # Restore backup
                shutil.copy2(backup_path, ch1_path)
                bad += 1
                kept_gt.append(subj)
            else:
                match_str = f"[{match_type}]" if match_type != "exact" else ""
                print(f"  OK  {subj:<55} labels={labels} {match_str}")
                ok += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Channel 1 Update Summary")
    print(f"  Updated successfully: {ok}/{total}")
    print(f"  Missing predictions:  {missing}")
    print(f"  Bad/skipped:          {bad}")
    print(f"  Fuzzy match warnings: {fuzzy_warn}")

    if kept_gt:
        print(f"\n  TRAIN/INFERENCE MISMATCH RISK:")
        print(f"  {len(kept_gt)} subjects kept GT coarse as channel 1:")
        for s in kept_gt:
            print(f"    {s}")
        print(f"  These subjects see GT coarse at training time but model")
        print(f"  predictions at inference time. This will reduce Stage 2 Dice.")
        print(f"  Resolve by ensuring Stage 1 cross-val preds cover all subjects.")

    if ok == total:
        print(f"\n  All channel 1 files updated successfully.")
        print(f"  Original GT coarse files backed up to: {BACKUP_DIR}")
        print(f"\n  NEXT STEP:")
        print(f"    nnUNetv2_preprocess -d 202 -plans_name nnUNetResEncUNetMPlans "
              f"-c 3d_fullres --verify_dataset_integrity")
    print("=" * 65)


if __name__ == "__main__":
    main()
