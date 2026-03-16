#!/bin/bash
# SLURM array job: Gemma 2B attack experiments — all defenses × 4 tasks × 4 attackers
# Submit AFTER slurm_train.sh completes (all GEMMA models must exist).
# Submit with: sbatch slurm_attack_gemma.sh
#
# Uses RTX 6000 (24GB) or L40S (48GB) for faster Gemma inference.
# 28 defenses × 4 attackers × 4 tasks = 448 jobs (array 0-447)
# Index: i = task_idx*112 + attacker_idx*28 + defense_idx

#SBATCH -J bodega_gemma
#SBATCH -p high
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=14:00
#SBATCH --array=0-447
#SBATCH -o logs/attack_gemma_%A_%a.out
#SBATCH -e logs/attack_gemma_%A_%a.err

TASKS=(PR2 FC HN RD)
ATTACKERS=(BERTattack PWWS DeepWordBug Genetic)

# 28 (defense, param) pairs — must match run_experiment-7_bilstm.sh exactly
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

# Gemma output is a directory (PEFT adapter), not a .pth file
MODEL_PATH="data/$TASK/GEMMA-512"
OUT_DIR="results/experiment-7_multi"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
mkdir -p "$OUT_DIR" logs

echo "[$i] GEMMA | $TASK | $ATTACK | $DEFENSE (param=$PARAM)"

if [ "$DEFENSE" = "none" ]; then
    python runs/attack.py "$TASK" false "$ATTACK" GEMMA \
        "data/$TASK" "$MODEL_PATH" "$OUT_DIR" --defense none
else
    python runs/attack.py "$TASK" false "$ATTACK" GEMMA \
        "data/$TASK" "$MODEL_PATH" "$OUT_DIR" \
        --defense "$DEFENSE" --defense_param "$PARAM" --defense_seed 42
fi
