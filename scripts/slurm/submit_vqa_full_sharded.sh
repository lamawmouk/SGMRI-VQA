#!/bin/bash
# Submit sharded inference jobs for qwen3-vl-8b-mri-vqa-full on H200s
# Usage: bash evaluation/submit_vqa_full_sharded.sh

BASE="/storage/ice-shared/ae8803che/models/VLM_grounding/Grounding"
MODEL="qwen3-vl-8b-mri-vqa-full"
GPU=h200
MEM=140G
mkdir -p logs

submit_shard() {
    local DS=$1 MODE=$2 START=$3 END=$4 SHARD=$5
    local JOB_NAME="v4_${MODEL}_${DS}_${MODE}_s${SHARD}"
    local SUFFIX="_${DS}_s${START}-${END}"

    sbatch --job-name="$JOB_NAME" \
        --account=coc --partition=ice-gpu \
        --nodes=1 --ntasks-per-node=1 \
        --gres=gpu:${GPU}:1 --mem-per-gpu=${MEM} --time=16:00:00 --qos=coc-ice \
        --output="logs/${JOB_NAME}_%j.out" \
        --error="logs/${JOB_NAME}_%j.err" \
        --wrap="
source ~/.bashrc
conda activate vlm_eval
export PYTHONPATH=${BASE}:\$PYTHONPATH
export HF_HOME=/storage/ice-shared/ae8803che/models
export TRANSFORMERS_CACHE=/storage/ice-shared/ae8803che/models
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd ${BASE}
python -m evaluation.run_inference --models ${MODEL} --eval-mode ${MODE} --datasets ${DS} --save-every 20 --no-resume --output-suffix ${SUFFIX} --start-idx ${START} --end-idx ${END}
"
    echo "Submitted ${DS}/${MODE} shard ${SHARD} (${START}-${END}) on ${GPU}"
}

# --- IMAGE: resume knee (was 74% done, ~5274 total) ---
# Split remaining into 2 shards, but since we use --no-resume, split full range into 4
# brain image: ~4654 samples -> 4 shards of ~1164
submit_shard brain image 0    1164 0
submit_shard brain image 1164 2328 1
submit_shard brain image 2328 3492 2
submit_shard brain image 3492 4654 3

# knee image: ~5274 samples -> 4 shards of ~1319
submit_shard knee image 0    1319 0
submit_shard knee image 1319 2638 1
submit_shard knee image 2638 3957 2
submit_shard knee image 3957 5274 3

# --- VIDEO: much slower, split into 4 shards each ---
# brain video: ~1185 samples -> 4 shards of ~296
submit_shard brain video 0   296  0
submit_shard brain video 296 592  1
submit_shard brain video 592 888  2
submit_shard brain video 888 1185 3

# knee video: ~929 samples -> 4 shards of ~232
submit_shard knee video 0   232 0
submit_shard knee video 232 464 1
submit_shard knee video 464 696 2
submit_shard knee video 696 929 3

echo ""
echo "Total: 16 sharded jobs submitted on ${GPU} GPUs"
