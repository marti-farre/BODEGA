#!/bin/bash
# Find missing BLEURT result files and output array indices for resubmission
# Usage: bash scripts/find_missing_bleurt.sh [BiLSTM|BERT|GEMMA]

VICTIM="${1:-BiLSTM}"
OUT_DIR="results/experiment-7_bleurt"

TASKS=(PR2 FC HN RD)
ATTACKERS=(BERTattack PWWS DeepWordBug Genetic)
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

missing=()
total=0
found=0

for i in $(seq 0 447); do
    task_idx=$((i / 112))
    attacker_idx=$(( (i % 112) / 28 ))
    defense_idx=$((i % 28))

    TASK=${TASKS[$task_idx]}
    ATTACK=${ATTACKERS[$attacker_idx]}
    DEF_ENTRY=${DEFENSES[$defense_idx]}
    DEFENSE=${DEF_ENTRY%%:*}
    PARAM=${DEF_ENTRY##*:}

    # Build expected result filename (matches attack.py naming)
    if [ "$DEFENSE" = "none" ]; then
        RESULT_FILE="${OUT_DIR}/results_${TASK}_False_${ATTACK}_${VICTIM}.txt"
    elif [ "$PARAM" = "0" ]; then
        RESULT_FILE="${OUT_DIR}/results_${TASK}_False_${ATTACK}_${VICTIM}_${DEFENSE}.txt"
    else
        RESULT_FILE="${OUT_DIR}/results_${TASK}_False_${ATTACK}_${VICTIM}_${DEFENSE}_${PARAM}.txt"
    fi

    total=$((total + 1))
    if [ -f "$RESULT_FILE" ]; then
        found=$((found + 1))
    else
        missing+=($i)
    fi
done

echo "$VICTIM: $found/$total results found, $((total - found)) missing"

if [ ${#missing[@]} -gt 0 ]; then
    # Format as SLURM array spec
    echo "Missing indices: ${missing[*]}"
    # Create comma-separated list for --array
    ARRAY_SPEC=$(IFS=,; echo "${missing[*]}")
    echo ""
    echo "Resubmit command:"
    SCRIPT_NAME="slurm_bleurt_$(echo $VICTIM | tr '[:upper:]' '[:lower:]').sh"
    echo "sbatch --array=$ARRAY_SPEC scripts/$SCRIPT_NAME"
fi
