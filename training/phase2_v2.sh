#!/bin/bash
#SBATCH --job-name=phase2_v2
#SBATCH --output=/scratch/users/jfundaun/bpseg/logs/phase2_v2_%j.log
#SBATCH --error=/scratch/users/jfundaun/bpseg/logs/phase2_v2_%j.err
#SBATCH --partition=gpu
#SBATCH --account=smackey
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
export nnUNet_raw="/scratch/users/jfundaun/bpseg/nnunet_cascade/raw"
export nnUNet_preprocessed="/scratch/users/jfundaun/bpseg/nnunet_cascade/preprocessed"
export nnUNet_results="/scratch/users/jfundaun/bpseg/nnunet_cascade/results"

RESULTS_201="${nnUNet_results}/Dataset201_DRGPlexusCoarse/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
CROSSVAL_DIR="/scratch/users/jfundaun/bpseg/derivatives/stage1_crossval_preds"
MERGED_DIR="${CROSSVAL_DIR}/merged"
SCRIPTS_DIR="/scratch/users/jfundaun/bpseg/scripts/final_complete_18april2026"
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

# Clear stale cross-val predictions from the PREVIOUS Dataset201 model so
# crossval_preds.py regenerates them fresh using the newly-retrained
# (324-case) Dataset201 model.
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
BACKUP_DIR="/scratch/users/jfundaun/bpseg/nnunet_cascade/preprocessed_backups"
mkdir -p "$BACKUP_DIR"

# FIX vs original phase2.sh: that version renamed the ENTIRE
# Dataset202_DRGPlexusFine preprocessed directory to
# "Dataset202_DRGPlexusFine_backup_<ts>", which creates a SECOND
# folder under preprocessed/ whose name also starts with "Dataset202_".
# nnUNet's convert_id_to_dataset_name(202) then finds two different
# names for the same dataset ID and raises:
#   "More than one dataset name found for dataset id 202"
#
# This version instead moves ONLY the per-case preprocessed-data dir,
# gt_segmentations, and splits_final.json out to a separate backups
# directory -- leaving nnUNetResEncUNetMPlans.json /
# dataset_fingerprint.json / dataset.json in place, which
# nnUNetv2_preprocess (via PlansManager) requires to already exist.
# Deleting splits_final.json forces a fresh 5-fold split covering all
# current training cases.
for sub in nnUNetPlans_3d_fullres nnUNetResEncUNetMPlans_3d_fullres gt_segmentations splits_final.json; do
    if [ -e "${PREPROC_202}/${sub}" ]; then
        mv "${PREPROC_202}/${sub}" "${BACKUP_DIR}/Dataset202_${sub}_${TS}"
        echo "  backed up ${sub} -> ${BACKUP_DIR}/Dataset202_${sub}_${TS}"
    fi
done


nnUNetv2_preprocess -d 202 -plans_name nnUNetResEncUNetMPlans -c 3d_fullres --verify_dataset_integrity

echo "=== Phase 2 (v2) complete: $(date) ==="
