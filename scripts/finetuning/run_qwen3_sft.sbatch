#!/bin/bash
#SBATCH --job-name=qwen3_sft_8h200
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-grade
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/ice-shared/ae8003/models/logs/qwen3_sft_%j.log
#SBATCH --error=/storage/ice-shared/ae8003/models/logs/qwen3_sft_%j.err

# Load modules and activate env
source ~/.bashrc
enter_project_vlm
export TRANSFORMERS_VIDEO_BACKEND="av"
export WANDB_PROJECT=vlm-finetuning

# Training mode: "full" (default) or "lora"
MODE="${1:-full}"

# Start nvitop monitoring in the background
/storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/.venv/bin/nvitop --mode monitor --interval 5 --logger-format text --logger-file /storage/ice-shared/ae8003/models/logs/gpu_usage_%j.txt &
MONITOR_PID=$!

# Run the appropriate SFT script
cd /storage/ice-shared/ae8803che/hxue/data/eccv/VideoMRIVQA/finetuneqwen/
if [ "$MODE" = "lora" ]; then
    echo "Running LoRA training..."
    ./qwen3_vl_8b_brains_combined.sh
else
    echo "Running full SFT training..."
    ./qwen3_vl_8b_brains_combined_full.sh
fi

# Kill monitoring after training
kill $MONITOR_PID
