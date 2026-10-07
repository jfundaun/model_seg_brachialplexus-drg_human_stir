import json, os, subprocess, glob, shutil, sys
import nibabel as nib
import numpy as np

nnUNet_raw          = os.environ["nnUNet_raw"]
nnUNet_preprocessed = os.environ["nnUNet_preprocessed"]

SPLITS_F = f"{nnUNet_preprocessed}/Dataset201_DRGPlexusCoarse/splits_final.json"
IMG_DIR  = f"{nnUNet_raw}/Dataset201_DRGPlexusCoarse/imagesTr"
IMG_TS   = f"{nnUNet_raw}/Dataset201_DRGPlexusCoarse/imagesTs"
OUT_BASE = "/.../derivatives/stage1_crossval_preds"
FINAL    = f"{OUT_BASE}/merged"

available_folds = [int(f) for f in sys.argv[1].split()]
print(f"Available folds: {available_folds}")

if not os.path.exists(SPLITS_F):
    print(f"ERROR: splits_final.json not found: {SPLITS_F}")
    sys.exit(1)

splits = json.load(open(SPLITS_F))
os.makedirs(FINAL, exist_ok=True)

def run_predict(input_dir, output_dir, folds_to_use):
    os.makedirs(output_dir, exist_ok=True)
    existing = glob.glob(f"{output_dir}/*.nii.gz")
    n_inputs = len(glob.glob(f"{input_dir}/*_0000.nii.gz"))
    if len(existing) >= n_inputs > 0:
        print(f"  Already predicted ({len(existing)} files) -- skipping")
        return
    cmd = ["nnUNetv2_predict", "-d", "201",
           "-i", input_dir, "-o", output_dir,
           "-f", *[str(f) for f in folds_to_use],
           "-c", "3d_fullres", "-tr", "nnUNetTrainer",
           "-p", "nnUNetResEncUNetMPlans"]
    print(f"  CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def stage_cases(case_ids, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    staged = 0
    for cid in case_ids:
        src = f"{src_dir}/{cid}_0000.nii.gz"
        dst = f"{dst_dir}/{cid}_0000.nii.gz"
        if os.path.exists(src):
            if not os.path.exists(dst):
                os.symlink(src, dst)
            staged += 1
        else:
            print(f"  WARNING: image not found: {src}")
    return staged

# Cross-val: predict each fold's val cases using all other available folds
for fold_k in range(len(splits)):
    val_cases = splits[fold_k]["val"]
    predict_with = [f for f in available_folds if f != fold_k] if fold_k in available_folds else available_folds
    if not predict_with:
        print(f"WARNING: no folds available to predict fold {fold_k} -- skipping")
        continue
    print(f"\nFold {fold_k} val: {len(val_cases)} cases -> predict with folds {predict_with}")
    fold_in  = f"{OUT_BASE}/fold_{fold_k}_input"
    fold_out = f"{OUT_BASE}/fold_{fold_k}_output"
    staged = stage_cases(val_cases, IMG_DIR, fold_in)
    print(f"  Staged {staged}/{len(val_cases)} images")
    run_predict(fold_in, fold_out, predict_with)
    for p in glob.glob(f"{fold_out}/*.nii.gz"):
        shutil.copy2(p, FINAL)
    print(f"  Copied {len(glob.glob(f'{fold_out}/*.nii.gz'))} preds to merged")

# Test set — all available folds
n_ts = len(glob.glob(f"{IMG_TS}/*_0000.nii.gz"))
print(f"\nTest set ({n_ts} cases) -> predict with folds {available_folds}")
test_out = f"{OUT_BASE}/test_output"
run_predict(IMG_TS, test_out, available_folds)
for p in glob.glob(f"{test_out}/*.nii.gz"):
    shutil.copy2(p, FINAL)

# Integrity check
preds = sorted(glob.glob(f"{FINAL}/*.nii.gz"))
print(f"\nVerifying {len(preds)} merged predictions...")
n_bad = 0
for p in preds:
    d   = nib.load(p).get_fdata(dtype=np.float32)
    err = float(np.abs(d - np.round(d)).max())
    mx  = int(d.max()) if d.size > 0 else 0
    if err > 1e-4 or mx > 2:
        print(f"  BAD: {os.path.basename(p)}  err={err:.8f}  max_label={mx}")
        n_bad += 1
if n_bad > 0:
    print(f"ERROR: {n_bad} predictions failed integrity check")
    sys.exit(1)
print(f"All {len(preds)} predictions verified clean.")
