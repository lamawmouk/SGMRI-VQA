#!/bin/bash
# Submit inference + evaluation jobs for qwen3-vl-8b-mri-vqa-full
# Usage: bash evaluation/submit_vqa_full_eval.sh

BASE="/storage/ice-shared/ae8803che/models/VLM_grounding/Grounding"
MODEL="qwen3-vl-8b-mri-vqa-full"
mkdir -p logs

submit_inference() {
    local EVAL_MODE=$1
    local GPU_TYPE=${2:-l40s}
    local MEM=${3:-80G}
    local TIME=${4:-12:00:00}

    JOB_NAME="v4_${MODEL}_${EVAL_MODE}"

    sbatch --job-name="$JOB_NAME" \
        --account=coc \
        --partition=ice-gpu \
        --nodes=1 --ntasks-per-node=1 \
        --gres=gpu:${GPU_TYPE}:1 \
        --mem-per-gpu=${MEM} \
        --time=${TIME} \
        --qos=coc-ice \
        --output="logs/v4_${MODEL}_${EVAL_MODE}_%j.out" \
        --error="logs/v4_${MODEL}_${EVAL_MODE}_%j.err" \
        --wrap="
source ~/.bashrc
conda activate vlm_eval
export PYTHONPATH=${BASE}:\$PYTHONPATH
export HF_HOME=/storage/ice-shared/ae8803che/models
export TRANSFORMERS_CACHE=/storage/ice-shared/ae8803che/models
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd ${BASE}
python -m evaluation.run_inference --models ${MODEL} --eval-mode ${EVAL_MODE} --save-every 20 --no-resume
"
    echo "Submitted ${MODEL} ${EVAL_MODE} (${GPU_TYPE}, ${TIME})"
}

# Submit image and video inference
submit_inference image l40s 80G 12:00:00
submit_inference video l40s 80G 08:00:00

echo ""
echo "2 jobs submitted for ${MODEL} (image + video)"
echo ""
echo "After inference completes, run evaluation with:"
echo "  python -m evaluation.run_evaluation --models ${MODEL} --eval-mode all"
