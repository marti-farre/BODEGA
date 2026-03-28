#!/bin/bash
# SLURM array job: BERT attack experiments with BLEURT scorer
# All defenses × 4 tasks × 4 attackers = 448 jobs (array 0-447)
# Index: i = task_idx*112 + attacker_idx*28 + defense_idx
#
# Submit with: cd ~/BODEGA && sbatch scripts/slurm_bleurt_bert.sh

#SBATCH -J bleurt_bert
#SBATCH -p medium
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --array=0-447
#SBATCH -o logs/bleurt_bert_%A_%a.out
#SBATCH -e logs/bleurt_bert_%A_%a.err

TASKS=(PR2 FC HN RD)
ATTACKERS=(BERTattack PWWS DeepWordBug Genetic)

# 28 (defense, param) pairs
DEFENSES=(
    none:0
    spellcheck:0
    char_noise:0.05  char_noise:0.10  char_noise:0.15  char_noise:0.20
    char_masking:0.05 char_masking:0.10 char_masking:0.15 char_masking:0.20
    unicode:0
    majority_vote:3  majority_vote:5  majority_vote:7
    label_flip:0.05  label_flip:0.10  label_flip:0.15
    random_threshold:0.05 random_threshold:0.10 random_threshold:0.15
    confidence_noise:0.05 confidence_noise:0.10 confidence_noise:0.15
    discretize:0
    spellcheck_mv:3  spellcheck_mv:7
    unicode_mv:3     unicode_mv:7
)

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$((i / 112))]}
ATTACK=${ATTACKERS[$(( (i % 112) / 28 ))]}
DEF_ENTRY=${DEFENSES[$((i % 28))]}
DEFENSE=${DEF_ENTRY%%:*}
PARAM=${DEF_ENTRY##*:}

MODEL_PATH="data/$TASK/BERT-512.pth"
OUT_DIR="results/experiment-7_bleurt"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
mkdir -p "$OUT_DIR" logs

echo "[$i] BERT+BLEURT | $TASK | $ATTACK | $DEFENSE (param=$PARAM)"

if [ "$DEFENSE" = "none" ]; then
    python runs/attack.py "$TASK" false "$ATTACK" BERT \
        "data/$TASK" "$MODEL_PATH" "$OUT_DIR" \
        --defense none --semantic_scorer BLEURT
else
    python runs/attack.py "$TASK" false "$ATTACK" BERT \
        "data/$TASK" "$MODEL_PATH" "$OUT_DIR" \
        --defense "$DEFENSE" --defense_param "$PARAM" --defense_seed 42 \
        --semantic_scorer BLEURT
fi
