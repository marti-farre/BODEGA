#!/bin/bash
# SLURM array job: BiLSTM HN-only with BLEURT + 500 query limit
# HN = task index 2, so array indices 224-335 in the full matrix
# But here we use local index 0-111 mapped to HN only
#
# Submit with: cd ~/BODEGA && sbatch scripts/slurm_bleurt_hn_bilstm.sh

#SBATCH -J hn_bilstm
#SBATCH -p medium
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-111
#SBATCH -o logs/hn_bilstm_%A_%a.out
#SBATCH -e logs/hn_bilstm_%A_%a.err

TASK="HN"
ATTACKERS=(BERTattack PWWS DeepWordBug Genetic)
MAX_QUERIES=500

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
ATTACK=${ATTACKERS[$((i / 28))]}
DEF_ENTRY=${DEFENSES[$((i % 28))]}
DEFENSE=${DEF_ENTRY%%:*}
PARAM=${DEF_ENTRY##*:}

MODEL_PATH="data/$TASK/BiLSTM-512.pth"
OUT_DIR="results/experiment-7_bleurt"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
mkdir -p "$OUT_DIR" logs

echo "[$i] BiLSTM+BLEURT+LIMITED | $TASK | $ATTACK | $DEFENSE (param=$PARAM) | max_queries=$MAX_QUERIES"

if [ "$DEFENSE" = "none" ]; then
    python runs/attack.py "$TASK" false "$ATTACK" BiLSTM \
        "data/$TASK" "$MODEL_PATH" "$OUT_DIR" \
        --defense none --semantic_scorer BLEURT --max_queries $MAX_QUERIES
else
    python runs/attack.py "$TASK" false "$ATTACK" BiLSTM \
        "data/$TASK" "$MODEL_PATH" "$OUT_DIR" \
        --defense "$DEFENSE" --defense_param "$PARAM" --defense_seed 42 \
        --semantic_scorer BLEURT --max_queries $MAX_QUERIES
fi
