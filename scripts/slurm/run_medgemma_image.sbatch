#!/bin/bash
#SBATCH --job-name=medgemma_img
#SBATCH --output=/home/hice1/zzhao465/scratch/VLM_grounding/slurm_logs/medgemma_img_%j.out
#SBATCH --error=/home/hice1/zzhao465/scratch/VLM_grounding/slurm_logs/medgemma_img_%j.err
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --qos=coe-ice
#SBATCH --account=coc

source ~/.bashrc
conda activate vlm_eval
cd /home/hice1/zzhao465/scratch/VLM_grounding/Grounding

python -m evaluation.run_inference \
    --models medgemma-4b \
    --eval-mode image \
    --max-cases 25 --max-per-task 1 \
    --save-every 10 \
    --rate-limit-delay 0
