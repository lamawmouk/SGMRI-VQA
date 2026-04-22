#!/bin/bash
# Submit all v4 inference jobs — full dataset, all models
# Usage: bash evaluation/submit_v4_all.sh

BASE="/storage/ice-shared/ae8803che/models/VLM_grounding/Grounding"
mkdir -p logs

submit_job() {
    local MODEL=$1
    local EVAL_MODE=$2
    local GPU_TYPE=${3:-l40s}
    local MEM=${4:-80G}
    local TIME=${5:-12:00:00}

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

# Image-only models (no video support)
submit_job llava-med-v1.5  image l40s 80G 12:00:00
submit_job medgemma-4b     image l40s 80G 12:00:00

# Image + Video models
for MODEL in qwen3-vl-8b qwen2.5-vl-7b llava-video-7b internvl2.5-8b eagle2.5-8b qwen3-vl-8b-mri qwen3-vl-8b-mri-vqa-full; do
    submit_job "$MODEL" image l40s 80G 12:00:00
    submit_job "$MODEL" video l40s 80G 08:00:00
done

echo ""
echo "Total: 16 jobs submitted (9 image + 7 video)"
echo "Gemini-2.5-pro runs locally (API) — not submitted here"
