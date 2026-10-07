#!/bin/bash
#SBATCH --job-name=phase2_v2
#SBATCH --output=/.../logs/phase2_v2_%j.log
#SBATCH --error=/.../logs/phase2_v2_%j.err
#SBATCH --partition=gpu
#SBATCH --account=X
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
export nnUNet_raw="/.../nnunet_cascade/raw"
export nnUNet_preprocessed="/.../nnunet_cascade/preprocessed"
export nnUNet_results="/.../nnunet_cascade/results"

RESULTS_201="${nnUNet_results}/Dataset201_DRGPlexusCoarse/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
CROSSVAL_DIR="/.../derivatives/stage1_crossval_preds"
MERGED_DIR="${CROSSVAL_DIR}/merged"
SCRIPTS_DIR="/X "
MIN_FOLDS=3

echo "=== Phase 2 (v2) start: $(date) ==="

VALID_FOLDS=()
for fold in 0 1 2 3 4; do
    ckpt="${RESULTS_201}/fold_${fold}/checkpoint_final.pth"
    if [ -f "$ckpt" ]; then
        python3 -c "import torch,sys; torch.load('${ckpt}',map_location='cpu',weights_only=False); sys.exit(0)" 2>/dev/null \
            && VALID_FOLDS+=("$fold") || echo "fold_${fold}: corrupt"
    fi
done
echo "Valid folds: ${VALID_FOLDS[*]:-none}"
[ "${#VALID_FOLDS[@]}" -ge "$MIN_FOLDS" ] || { echo "ERROR: only ${#VALID_FOLDS[@]} valid folds"; exit 1; }
FOLDS_STR="${VALID_FOLDS[*]}"

# Dataset201 model.
if [ -d "$MERGED_DIR" ]; then
    mv "$MERGED_DIR" "${MERGED_DIR}_stale_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$MERGED_DIR"

python3 "${SCRIPTS_DIR}/crossval_preds.py" "$FOLDS_STR"

echo "Updating channel 1..."
python3 "${SCRIPTS_DIR}/update_channel1.py"

echo "Preprocessing Dataset202..."
PREPROC_202="${nnUNet_preprocessed}/Dataset202_DRGPlexusFine"
TS=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/.../nnunet_cascade/preprocessed_backups"
mkdir -p "$BACKUP_DIR"

for sub in nnUNetPlans_3d_fullres nnUNetResEncUNetMPlans_3d_fullres gt_segmentations splits_final.json; do
    if [ -e "${PREPROC_202}/${sub}" ]; then
        mv "${PREPROC_202}/${sub}" "${BACKUP_DIR}/Dataset202_${sub}_${TS}"
        echo "  backed up ${sub} -> ${BACKUP_DIR}/Dataset202_${sub}_${TS}"
    fi
done


nnUNetv2_preprocess -d 202 -plans_name nnUNetResEncUNetMPlans -c 3d_fullres --verify_dataset_integrity

echo "=== Phase 2 (v2) complete: $(date) ==="
