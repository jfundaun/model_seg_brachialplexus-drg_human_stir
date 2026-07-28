#!/bin/bash
#SBATCH --job-name=train_202
#SBATCH --output=/scratch/users/jfundaun/bpseg/logs/train_202_f%a_%j.log
#SBATCH --error=/scratch/users/jfundaun/bpseg/logs/train_202_f%a_%j.err
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --account=smackey
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --constraint="GPU_GEN:AMP"

set -euo pipefail
FOLD=${SLURM_ARRAY_TASK_ID}

export nnUNet_raw="/scratch/users/jfundaun/bpseg/nnunet_cascade/raw"
export nnUNet_preprocessed="/scratch/users/jfundaun/bpseg/nnunet_cascade/preprocessed"
export nnUNet_results="/scratch/users/jfundaun/bpseg/nnunet_cascade/results"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Training Dataset202 fold ${FOLD} on ${GPU} at $(date) on $(hostname)"

NNUNET_COMPILE=0 nnUNetv2_train 202 3d_fullres ${FOLD} \
    -tr nnUNetTrainer \
    -p nnUNetResEncUNetMPlans

echo "Dataset202 fold ${FOLD} COMPLETE at $(date)"

# Ensure gt_segmentations exists for post-training validation
GT_DIR="${nnUNet_preprocessed}/Dataset202_DRGPlexusFine/gt_segmentations"
mkdir -p "${GT_DIR}"
N=$(ls "${GT_DIR}" 2>/dev/null | wc -l)
if [ "${N}" -eq 0 ]; then
    echo "Copying Dataset202 GT segmentations for validation..."
    cp "${nnUNet_raw}/Dataset202_DRGPlexusFine/labelsTr/"*.nii.gz "${GT_DIR}/"
fi
