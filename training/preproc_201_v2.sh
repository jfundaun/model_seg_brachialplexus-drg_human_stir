#!/bin/bash
#SBATCH --job-name=preproc_201_v2
#SBATCH --output=/scratch/users/jfundaun/bpseg/logs/preproc_201_v2_%j.log
#SBATCH --error=/scratch/users/jfundaun/bpseg/logs/preproc_201_v2_%j.err
#SBATCH --partition=gpu
#SBATCH --account=smackey
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
export nnUNet_raw="/scratch/users/jfundaun/bpseg/nnunet_cascade/raw"
export nnUNet_preprocessed="/scratch/users/jfundaun/bpseg/nnunet_cascade/preprocessed"
export nnUNet_results="/scratch/users/jfundaun/bpseg/nnunet_cascade/results"

DS_DIR="${nnUNet_preprocessed}/Dataset201_DRGPlexusCoarse"
RAW_JSON="${nnUNet_raw}/Dataset201_DRGPlexusCoarse/dataset.json"
SPLITS="${DS_DIR}/splits_final.json"

N_RAW=$(python3 -c "import json; print(json.load(open('${RAW_JSON}'))['numTraining'])")

if [ -f "$SPLITS" ]; then
    N_SPLITS=$(python3 -c "
import json
s = json.load(open('${SPLITS}'))
cases = set()
for f in s:
    cases.update(f['train']); cases.update(f['val'])
print(len(cases))
")
else
    N_SPLITS=0
fi

echo "raw dataset.json numTraining = ${N_RAW}"
echo "splits_final.json unique cases = ${N_SPLITS}"

if [ -f "$SPLITS" ] && [ "$N_RAW" -eq "$N_SPLITS" ]; then
    echo "Preprocessed split already matches raw case count -- skipping preprocessing."
    exit 0
fi

if [ -f "$SPLITS" ]; then
    echo "Case count mismatch (raw=${N_RAW} vs splits=${N_SPLITS})."
    echo "Removing stale splits_final.json so nnUNet regenerates a fresh split"
    echo "covering all ${N_RAW} training cases."
    rm -f "$SPLITS"
fi

echo "Preprocessing Dataset201..."
nnUNetv2_preprocess -d 201 -plans_name nnUNetResEncUNetMPlans -c 3d_fullres --verify_dataset_integrity
echo "Dataset201 preprocessing COMPLETE at $(date)"
