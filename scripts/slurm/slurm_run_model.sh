#!/bin/bash
#SBATCH --job-name=vlm_eval
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%x_%j_%a.out
#SBATCH --error=slurm_logs/%x_%j_%a.err

# Usage:
#   sbatch slurm_run_model.sh MODEL_KEY [EVAL_MODE]
#   sbatch slurm_run_model.sh qwen2.5-vl-7b image
#   sbatch slurm_run_model.sh qwen2.5-vl-7b video
#   sbatch slurm_run_model.sh qwen2.5-vl-7b all    # runs both image and video
#
# Or run interactively after srun:
#   bash slurm_run_model.sh MODEL_KEY [EVAL_MODE]

MODEL_KEY="${1:?Usage: sbatch slurm_run_model.sh MODEL_KEY [EVAL_MODE]}"
EVAL_MODE="${2:-all}"

REMOTE_DIR="/storage/ice1/8/4/zzhao465/VLM_grounding"
cd "$REMOTE_DIR/Grounding"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate vlm_eval

# Load CUDA module if needed
module load cuda/12.1.1 2>/dev/null || true

echo "============================================"
echo "Model: $MODEL_KEY"
echo "Eval mode: $EVAL_MODE"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

# Set env vars for Qwen3-Omni
if [ "$MODEL_KEY" = "qwen3-omni" ]; then
    export VLLM_USE_V1=0
fi

# HuggingFace cache on shared storage (no quota issues)
export HF_HOME="/storage/ice-shared/ae8803che/models"
export TRANSFORMERS_CACHE="/storage/ice-shared/ae8803che/models"

# Skip flash_attn — compilation takes hours and blocks jobs.
# Models fall back to sdpa/eager attention automatically.
python -c "import flash_attn; print('flash_attn available')" 2>/dev/null || {
    echo ">>> flash_attn not available — models will use sdpa/eager fallback"
}

run_inference() {
    local mode=$1
    echo ""
    echo ">>> Running $MODEL_KEY in $mode mode..."
    echo ">>> $(date)"
    python -m evaluation.run_inference \
        --models "$MODEL_KEY" \
        --eval-mode "$mode" \
        --save-every 20
    echo ">>> Finished $MODEL_KEY $mode at $(date)"
}

if [ "$EVAL_MODE" = "all" ]; then
    run_inference image
    run_inference video
elif [ "$EVAL_MODE" = "image" ] || [ "$EVAL_MODE" = "video" ]; then
    run_inference "$EVAL_MODE"
else
    echo "ERROR: Invalid eval mode '$EVAL_MODE'. Use: image, video, or all"
    exit 1
fi

echo ""
echo "============================================"
echo "All done for $MODEL_KEY at $(date)"
echo "============================================"
