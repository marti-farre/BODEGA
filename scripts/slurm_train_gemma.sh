#!/bin/bash
# SLURM array job: train Gemma 2B for all 4 tasks (PR2, FC, HN, RD)
# Requires RTX 6000 (24GB) or L40S (48GB) — T4 (10.9GB) is too small for Gemma.
# Submit with: sbatch scripts/slurm_train_gemma.sh
# Array index mapping: 0→PR2, 1→FC, 2→HN, 3→RD

#SBATCH -J bodega_gemma_train
#SBATCH -p high
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=14:00:00
#SBATCH --array=0-3
#SBATCH -o logs/train_gemma_%A_%a.out
#SBATCH -e logs/train_gemma_%A_%a.err

TASKS=(PR2 FC HN RD)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
MODEL_PATH="data/$TASK/GEMMA-512"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
HF_TOKEN_VALUE=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export HF_TOKEN=$HF_TOKEN_VALUE
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN_VALUE
echo "HF token loaded: ${HF_TOKEN_VALUE:0:8}..."

echo "Training GEMMA on $TASK → $MODEL_PATH"
python runs/train_victims.py "$TASK" GEMMA "data/$TASK" "$MODEL_PATH"
