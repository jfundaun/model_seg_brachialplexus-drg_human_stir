#!/usr/bin/env python3
"""
Cascaded nnUNet Data Preparation for DRG + Brachial Plexus Segmentation

Usage:
  # Initial prep (GT coarse labels as channel 1):
  python prepare_nnunet_data.py

  # Final prep (Stage 1 predictions as channel 1):
  python prepare_nnunet_data.py --dataset202_only \
      --stage1_pred_dir /path/to/crossval_preds
"""

import os, sys, shutil, json, argparse
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.ndimage import map_coordinates

CSV_PATH    = "/.../bpseg.csv"
NNUNET_BASE = "/.../nnunet_cascade"

DATASET_201_NAME = "Dataset201_DRGPlexusCoarse"
DATASET_202_NAME = "Dataset202_DRGPlexusFine"

DRG_LABELS    = list(range(1, 9))
PLEXUS_LABELS = list(range(9, 17))
IGNORE_LABELS = list(range(17, 21))

COARSE_MAP = {0:0, **{i:1 for i in DRG_LABELS},
              **{i:2 for i in PLEXUS_LABELS}, **{i:0 for i in IGNORE_LABELS}}

FINE_LABEL_NAMES = {
    "background":0, "DRG_C5_R":1, "DRG_C6_R":2, "DRG_C7_R":3, "DRG_C8_R":4,
    "DRG_C5_L":5,   "DRG_C6_L":6, "DRG_C7_L":7, "DRG_C8_L":8,
    "BP_C5_R":9,    "BP_C6_R":10, "BP_C7_R":11, "BP_C8_R":12,
    "BP_C5_L":13,   "BP_C6_L":14, "BP_C7_L":15, "BP_C8_L":16,
}

def make_dirs(p): os.makedirs(p, exist_ok=True); return p

def get_subject_id(vol_path):
    stem = Path(vol_path).name.replace(".nii.gz","").replace(".nii","")
    return stem.replace(" ","_").replace("-","_")

def affines_match(a1, a2, tol=1e-3): return np.allclose(a1, a2, atol=tol)

def resample_label_to_ref(label_img, ref_img):
    src_affine = label_img.affine
    ref_affine = ref_img.affine
    ref_shape  = ref_img.shape[:3]
    if affines_match(src_affine, ref_affine) and label_img.shape[:3] == ref_shape:
        return label_img
    label_data = np.round(label_img.get_fdata()).astype(np.int32)
    i,j,k = np.meshgrid(np.arange(ref_shape[0]),np.arange(ref_shape[1]),
                        np.arange(ref_shape[2]),indexing='ij')
    ref_vox = np.stack([i.ravel(),j.ravel(),k.ravel(),np.ones(i.size)],axis=0)
    src_vox = (np.linalg.inv(src_affine) @ (ref_affine @ ref_vox))[:3,:]
    resampled = map_coordinates(label_data,src_vox,order=0,mode='constant',cval=0
                                ).reshape(ref_shape).astype(np.int32)
    return nib.Nifti1Image(resampled, ref_affine, ref_img.header)

def remap_to_coarse(d):
    c = np.zeros_like(d, dtype=np.uint8)
    for s,t in COARSE_MAP.items(): c[d==s]=t
    return c

def remap_fine(d):
    f = d.copy().astype(np.uint8)
    for l in IGNORE_LABELS: f[f==l]=0
    return f

def save_image_nifti(img, path): nib.save(img, path)

def save_seg_nifti(data, ref_img, path, dtype=np.uint8):
    new_hdr = ref_img.header.copy()
    new_hdr.set_data_dtype(dtype)
    new_hdr['scl_slope'] = 0.0   # 0 = disabled per NIfTI-1 spec
    new_hdr['scl_inter'] = 0.0
    new_hdr.set_qform(ref_img.affine, code=max(1,int(new_hdr['qform_code'])))
    new_hdr.set_sform(ref_img.affine, code=max(1,int(new_hdr['sform_code'])))
    nib.save(nib.Nifti1Image(data.astype(dtype), ref_img.affine, new_hdr), path)

def write_dataset_json(path, channel_names, labels, num_training, num_test=0):
    json.dump({"channel_names":channel_names,"labels":labels,
               "numTraining":num_training,"numTest":num_test,
               "file_ending":".nii.gz",
               "overwrite_image_reader_writer":"NibabelIOWithReorient"},
              open(path,"w"), indent=4)
    print(f"  Written: {path}")

def find_stage1_pred(sub_id, pred_dir):
    if not pred_dir: return None
    for pat in [f"{sub_id}.nii.gz", f"{sub_id}_seg.nii.gz",
                f"{sub_id}_seg_coarse.nii.gz"]:
        p = Path(pred_dir)/pat
        if p.exists(): return nib.load(str(p))
    hits = list(Path(pred_dir).glob(f"*{sub_id}*.nii.gz"))
    return nib.load(str(hits[0])) if hits else None

def process_case(vol_path, seg_path, sub_id, split, errors,
                 do_201=True, do_202=True, stage1_pred_dir=None):
    for p,n in [(vol_path,"image"),(seg_path,"label")]:
        if not os.path.exists(p):
            print(f"  WARNING: Missing {n}: {p}"); errors.append(p); return None

    vol_img = nib.load(vol_path)
    try:
        seg_img = nib.load(seg_path)
    except Exception as e:
        print(f"  WARNING: Cannot load seg {sub_id}: {e}"); errors.append(seg_path); return None

    seg_in_vol = resample_label_to_ref(seg_img, vol_img)
    if not affines_match(seg_img.affine, vol_img.affine):
        print(f"    Resampled seg?MRI for {sub_id}")
    seg_data  = np.round(seg_in_vol.get_fdata()).astype(np.int32)
    coarse_gt = remap_to_coarse(seg_data)
    fine_data = remap_fine(seg_data)

    if do_201:
        save_image_nifti(vol_img,
            f"{NNUNET_BASE}/raw/{DATASET_201_NAME}/images{split}/{sub_id}_0000.nii.gz")
        save_seg_nifti(coarse_gt, vol_img,
            f"{NNUNET_BASE}/raw/{DATASET_201_NAME}/labels{split}/{sub_id}.nii.gz")

    if do_202:
        save_image_nifti(vol_img,
            f"{NNUNET_BASE}/raw/{DATASET_202_NAME}/images{split}/{sub_id}_0000.nii.gz")
        save_seg_nifti(fine_data, vol_img,
            f"{NNUNET_BASE}/raw/{DATASET_202_NAME}/labels{split}/{sub_id}.nii.gz")

        s1pred = find_stage1_pred(sub_id, stage1_pred_dir)
        if s1pred is not None:
            s1_in_vol  = resample_label_to_ref(s1pred, vol_img)
            s1_data    = np.round(s1_in_vol.get_fdata()).astype(np.uint8)
            save_seg_nifti(s1_data, vol_img,
                f"{NNUNET_BASE}/raw/{DATASET_202_NAME}/images{split}/{sub_id}_0001.nii.gz")
            ch1 = f"stage1_pred"
        else:
            if stage1_pred_dir:
                print(f"  WARNING: Stage1 pred not found for {sub_id} - using GT coarse")
            save_seg_nifti(coarse_gt, vol_img,
                f"{NNUNET_BASE}/raw/{DATASET_202_NAME}/images{split}/{sub_id}_0001.nii.gz")
            ch1 = "GT_coarse"
        return coarse_gt, fine_data, ch1
    return coarse_gt, fine_data, "GT_coarse"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset202_only", action="store_true")
    parser.add_argument("--stage1_pred_dir", default=None)
    parser.add_argument("--csv", default=CSV_PATH)
    args = parser.parse_args()

    print("="*65)
    print("nnUNet Cascade Data Preparation")
    print(f"  Channel 1: {'Stage1 preds from '+args.stage1_pred_dir if args.stage1_pred_dir else 'GT coarse (fallback)'}")
    print("="*65)

    df = pd.read_csv(args.csv)
    df["subset"] = df["subset"].astype(str).str.strip().str.lower().replace("trains","train")
    print(f"Subsets: {df['subset'].value_counts().to_dict()}")

    df["nnunet_id"] = df["vol_path"].apply(get_subject_id)
    dupes = df[df["nnunet_id"].duplicated(keep=False)]
    if len(dupes):
        print(f"WARNING: {len(dupes)} duplicate nnUNet IDs - later rows overwrite!")
        print(dupes[["ID","subset","nnunet_id"]].to_string())

    trainval = pd.concat([df[df["subset"]=="train"], df[df["subset"]=="validation"]])
    test_df  = df[df["subset"]=="test"]

    datasets = [DATASET_202_NAME] if args.dataset202_only else [DATASET_201_NAME, DATASET_202_NAME]
    for ds in datasets:
        for sp in ["imagesTr","labelsTr","imagesTs","labelsTs"]:
            make_dirs(f"{NNUNET_BASE}/raw/{ds}/{sp}")

    errors = []; ch1_counts = {"stage1_pred":0,"GT_coarse":0}
    do_201 = not args.dataset202_only

    print(f"\nProcessing {len(trainval)} train/val cases...")
    for i,(_, row) in enumerate(trainval.iterrows()):
        sid = get_subject_id(row["vol_path"])
        r = process_case(row["vol_path"],row["seg_path"],sid,"Tr",errors,
                         do_201=do_201,stage1_pred_dir=args.stage1_pred_dir)
        if r:
            ch1_counts["stage1_pred" if r[2]=="stage1_pred" else "GT_coarse"] += 1
            print(f"  [{i+1:>3}/{len(trainval)}] {sid:<45} ch1={r[2]}")

    print(f"\nProcessing {len(test_df)} test cases...")
    for i,(_, row) in enumerate(test_df.iterrows()):
        sid = get_subject_id(row["vol_path"])
        process_case(row["vol_path"],row["seg_path"],sid,"Ts",errors,
                     do_201=do_201,stage1_pred_dir=args.stage1_pred_dir)
        print(f"  [test {i+1:>2}] {sid}")

    n_tr = len(list(Path(f"{NNUNET_BASE}/raw/{DATASET_201_NAME}/labelsTr").glob("*.nii.gz")))
    n_ts = len(list(Path(f"{NNUNET_BASE}/raw/{DATASET_201_NAME}/labelsTs").glob("*.nii.gz")))

    if do_201:
        write_dataset_json(f"{NNUNET_BASE}/raw/{DATASET_201_NAME}/dataset.json",
            {"0":"MRI"}, {"background":0,"DRG":1,"Plexus":2}, n_tr, n_ts)
    write_dataset_json(f"{NNUNET_BASE}/raw/{DATASET_202_NAME}/dataset.json",
        {"0":"MRI","1":"noNorm"}, FINE_LABEL_NAMES, n_tr, n_ts)

    print(f"\n{'='*65}")
    print(f"Done: {n_tr} train, {n_ts} test")
    print(f"Channel 1: {ch1_counts}")
    if errors: print(f"Errors: {errors}")
    print("="*65)

if __name__ == "__main__":
    main()
