#!/bin/bash
# SLURM array job: train BERT and Gemma 2B for all 4 tasks (PR2, FC, HN, RD)
# Submit with: sbatch slurm_train.sh
# Array index mapping: 0-3 → BERT × (PR2,FC,HN,RD), 4-7 → GEMMA × (PR2,FC,HN,RD)

#SBATCH -J bodega_train
#SBATCH -p medium
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --time=14:00
#SBATCH --array=0-7
#SBATCH -o logs/train_%A_%a.out
#SBATCH -e logs/train_%A_%a.err

TASKS=(PR2 FC HN RD PR2 FC HN RD)
VICTIMS=(BERT BERT BERT BERT GEMMA GEMMA GEMMA GEMMA)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
VICTIM=${VICTIMS[$SLURM_ARRAY_TASK_ID]}

# BERT → .pth file; GEMMA → directory (PEFT adapter)
if [ "$VICTIM" = "GEMMA" ]; then
    MODEL_PATH="data/$TASK/GEMMA-512"
else
    MODEL_PATH="data/$TASK/BERT-512.pth"
fi

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN

echo "Training $VICTIM on $TASK → $MODEL_PATH"
python runs/train_victims.py "$TASK" "$VICTIM" "data/$TASK" "$MODEL_PATH"
