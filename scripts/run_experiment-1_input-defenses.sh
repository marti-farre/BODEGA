#!/bin/bash
# =============================================================================
# Multi-attacker defense evaluation
# =============================================================================
#
# Tests all defense configurations against multiple attackers:
#   - BERTattack (word-level, BERT-based)
#   - PWWS (word-level, synonym replacement)
#   - DeepWordBug (character-level, typos)
#   - Genetic (word-level, genetic algorithm)
#
# Defense configurations (10 total):
#   - none (baseline)
#   - spellcheck
#   - char_masking at 0.05, 0.10, 0.15, 0.20
#   - char_noise at 0.05, 0.10, 0.15, 0.20
#
# Total experiments: 4 attackers x 10 defenses = 40 runs
#
# Usage:
#   ./scripts/run_multi_attacker_experiments.sh
#
# Prerequisites:
#   - Virtual environment activated
#   - Dependencies installed (including homoglyphs)
#   - PR2 data and model available
# =============================================================================

set -e  # Exit on error

# Set PYTHONPATH to find local modules
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

# Configuration
TASK="PR2"
TARGETED="false"
VICTIM="BiLSTM"
DATA_PATH="data/PR2"
MODEL_PATH="data/PR2/BiLSTM-512.pth"
OUT_DIR="results/experiment-1_add-noise-to-input"
DEFENSE_SEED=42

# Attackers to test
ATTACKERS=("BERTattack" "PWWS" "DeepWordBug" "Genetic")

# Defense configurations: "name:param"
declare -a DEFENSE_CONFIGS=(
    "none:0.0"
    "spellcheck:0.0"
    "char_masking:0.05"
    "char_masking:0.10"
    "char_masking:0.15"
    "char_masking:0.20"
    "char_noise:0.05"
    "char_noise:0.10"
    "char_noise:0.15"
    "char_noise:0.20"
)

# Create output directory
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Multi-Attacker Defense Evaluation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Task: $TASK"
echo "  Targeted: $TARGETED"
echo "  Victim: $VICTIM"
echo "  Data: $DATA_PATH"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUT_DIR"
echo ""
echo "Attackers: ${ATTACKERS[*]}"
echo "Defense configs: ${#DEFENSE_CONFIGS[@]}"
echo "Total experiments: $((${#ATTACKERS[@]} * ${#DEFENSE_CONFIGS[@]}))"
echo ""

# Count for progress
total_experiments=$((${#ATTACKERS[@]} * ${#DEFENSE_CONFIGS[@]}))
current=0

# =============================================================================
# Step 0: Evaluate clean accuracy (utility baseline)
# =============================================================================
echo "[0/$total_experiments] Evaluating clean accuracy (utility baseline)..."
python runs/eval_defense_accuracy.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR"
echo ""

# =============================================================================
# Run all attacker x defense combinations
# =============================================================================

for attacker in "${ATTACKERS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing attacker: $attacker"
    echo "=========================================="
    echo ""

    for config in "${DEFENSE_CONFIGS[@]}"; do
        # Parse defense name and parameter
        defense="${config%%:*}"
        param="${config##*:}"

        current=$((current + 1))

        # Build description
        if [ "$defense" == "none" ]; then
            desc="baseline (no defense)"
        elif [ "$param" == "0.0" ]; then
            desc="$defense"
        else
            desc="$defense (param=$param)"
        fi

        echo "[$current/$total_experiments] $attacker + $desc"

        # Run attack
        if [ "$defense" == "none" ]; then
            python runs/attack.py "$TASK" "$TARGETED" "$attacker" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense none
        elif [ "$param" == "0.0" ]; then
            python runs/attack.py "$TASK" "$TARGETED" "$attacker" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense "$defense"
        else
            python runs/attack.py "$TASK" "$TARGETED" "$attacker" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense "$defense" --defense_param "$param" --defense_seed "$DEFENSE_SEED"
        fi

        echo ""
    done
done

echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUT_DIR"
echo ""
echo "Result files:"
ls -la "$OUT_DIR"/results_*.txt 2>/dev/null | head -20 || echo "  (no result files found)"
echo ""
echo "To analyze results, check:"
echo "  - Individual results: $OUT_DIR/results_*.txt"
echo "  - Clean accuracy: $OUT_DIR/clean_accuracy_${TASK}_${VICTIM}.txt"
