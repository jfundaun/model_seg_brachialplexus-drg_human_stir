#!/bin/bash
# =============================================================
# Usage:
#   bash submit_pipeline.sh
# =============================================================

set -euo pipefail

cd /X

echo "=================================================="
echo "  Submitting full retraining pipeline"
echo "=================================================="

JOB_PREP=$(sbatch --parsable run_prepare_data.sh)
echo "  [0] run_prepare_data.sh   -> job ${JOB_PREP}"

JOB_PREPROC_201=$(sbatch --parsable \
    --dependency=afterok:${JOB_PREP} \
    preproc_201_v2.sh)
echo "  [1] preproc_201_v2.sh     -> job ${JOB_PREPROC_201}"
echo "        depends on: afterok:${JOB_PREP}"

JOB_TRAIN_201=$(sbatch --parsable \
    --dependency=afterok:${JOB_PREPROC_201} \
    train_201_fold.sh)
echo "  [2] train_201_fold.sh     -> job ${JOB_TRAIN_201} (array 0-4, 48h, AMP)"
echo "        depends on: afterok:${JOB_PREPROC_201}"

JOB_PHASE2=$(sbatch --parsable \
    --dependency=afterok:${JOB_TRAIN_201} \
    phase2_v2.sh)
echo "  [3] phase2_v2.sh          -> job ${JOB_PHASE2}"
echo "        depends on: afterok:${JOB_TRAIN_201} (all folds)"

JOB_TRAIN_202=$(sbatch --parsable \
    --dependency=afterok:${JOB_PHASE2} \
    train_202_fold.sh)
echo "  [4] train_202_fold.sh     -> job ${JOB_TRAIN_202} (array 0-4, 48h, AMP)"
echo "        depends on: afterok:${JOB_PHASE2}"

echo ""
echo "=================================================="
echo "  All jobs submitted."
echo ""
echo "  Job IDs:"
echo "    prepare_data   : ${JOB_PREP}"
echo "    preproc_201_v2 : ${JOB_PREPROC_201}"
echo "    train_201      : ${JOB_TRAIN_201}"
echo "    phase2_v2      : ${JOB_PHASE2}"
echo "    train_202      : ${JOB_TRAIN_202}"
echo ""
echo "  Monitor with:"
echo "    squeue --me"
echo "    sacct -j ${JOB_PREP},${JOB_PREPROC_201},${JOB_TRAIN_201},${JOB_PHASE2},${JOB_TRAIN_202} \\"
echo "      --format=JobID,JobName%20,State,ExitCode,Elapsed,Start,End"
echo "=================================================="
