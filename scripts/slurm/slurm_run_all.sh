#!/bin/bash
# Submit all 9 models as separate SLURM jobs
# Usage: bash slurm_run_all.sh [image|video|all]
#
# Each model gets its own job. Jobs run independently (can run in parallel
# if multiple GPUs are available, otherwise SLURM queues them).

EVAL_MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$SCRIPT_DIR/slurm_logs"

# Models with weights downloaded and ready
MODELS=(
    "qwen2.5-vl-7b"
    "qwen3-vl-8b"
    "internvl2.5-8b"
    "molmo2-8b"
    "keye-vl-1.5-8b"
    "llava-video-7b"
)
# Not ready yet (pending HF access):
#   "eagle2.5-8b"
#   "plm-8b"
# Skipped:
#   "qwen3-omni"

echo "Submitting ${#MODELS[@]} models with eval_mode=$EVAL_MODE"
echo ""

for model in "${MODELS[@]}"; do
    JOB_ID=$(sbatch --job-name="vlm_${model}" \
        --export=ALL \
        "$SCRIPT_DIR/slurm_run_model.sh" "$model" "$EVAL_MODE" \
        2>&1 | grep -oP '\d+')
    echo "  Submitted $model -> Job $JOB_ID"
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: slurm_logs/"
