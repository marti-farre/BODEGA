#!/bin/bash
# SLURM array job: Gemma with BLEURT — top 8 defenses only
# 8 defenses × 4 tasks × 4 attackers = 128 jobs (array 0-127)
# Index: i = task_idx*32 + attacker_idx*8 + defense_idx
#
# Submit with: cd ~/BODEGA && sbatch scripts/slurm_bleurt_gemma_top8.sh

#SBATCH -J gem_top8
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --array=0-127
#SBATCH -o logs/gem_top8_%A_%a.out
#SBATCH -e logs/gem_top8_%A_%a.err

TASKS=(PR2 FC HN RD)
ATTACKERS=(BERTattack PWWS DeepWordBug Genetic)
MAX_QUERIES=500

# Top 8 defenses (accuracy-penalized ranking from BiLSTM BLEURT results)
DEFENSES=(
    none:0
    discretize:0
    spellcheck:0
    label_flip:0.05
    confidence_noise:0.10
    spellcheck_mv:3
    spellcheck_mv:7
    majority_vote:7
)

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$((i / 32))]}
ATTACK=${ATTACKERS[$(( (i % 32) / 8 ))]}
DEF_ENTRY=${DEFENSES[$((i % 8))]}
DEFENSE=${DEF_ENTRY%%:*}
PARAM=${DEF_ENTRY##*:}

MODEL_PATH="data/$TASK/GEMMA-512"
OUT_DIR="results/experiment-7_bleurt"

source /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
conda activate bodega
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
mkdir -p "$OUT_DIR" logs

echo "[$i] GEMMA+BLEURT+TOP8 | $TASK | $ATTACK | $DEFENSE (param=$PARAM) | max_queries=$MAX_QUERIES"

if [ "$DEFENSE" = "none" ]; then
    python runs/attack.py "$TASK" false "$ATTACK" GEMMA \
        "data/$TASK" "$MODEL_PATH" "$OUT_DIR" \
        --defense none --semantic_scorer BLEURT --max_queries $MAX_QUERIES
else
    python runs/attack.py "$TASK" false "$ATTACK" GEMMA \
        "data/$TASK" "$MODEL_PATH" "$OUT_DIR" \
        --defense "$DEFENSE" --defense_param "$PARAM" --defense_seed 42 \
        --semantic_scorer BLEURT --max_queries $MAX_QUERIES
fi
