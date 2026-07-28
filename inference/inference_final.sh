#!/bin/bash
#SBATCH --job-name=inference_final
#SBATCH --output=/scratch/users/jfundaun/bpseg/logs/inference_final_%j.log
#SBATCH --error=/scratch/users/jfundaun/bpseg/logs/inference_final_%j.err
#SBATCH --partition=gpu
#SBATCH --account=smackey
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
export nnUNet_raw="/scratch/users/jfundaun/bpseg/nnunet_cascade/raw"
export nnUNet_preprocessed="/scratch/users/jfundaun/bpseg/nnunet_cascade/preprocessed"
export nnUNet_results="/scratch/users/jfundaun/bpseg/nnunet_cascade/results"

echo "Starting inference at $(date) on $(hostname)"

python3 /scratch/users/jfundaun/bpseg/scripts/final_complete_18april2026/inference_v2_fixed.py \
    --subset all \
    --output /scratch/users/jfundaun/bpseg/derivatives/cascade_results_final/23July2026

echo "Inference COMPLETE at $(date)"
