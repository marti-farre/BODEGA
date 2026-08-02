#!/bin/bash
# Experiments for survey-based defenses: Unicode Canonicalization and Majority Vote
# Branch: experiment-1/implement-survey-defenses

set -e

# Set PYTHONPATH to find local modules
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

# Configuration
TASK="PR2"
VICTIM="BiLSTM"
DATA_PATH="data/PR2"
MODEL_PATH="data/PR2/BiLSTM-512.pth"
# Results go in branch-specific subdirectory
OUT_DIR="results/experiment-1_implement-survey-defenses"
TARGETED="false"
SEED=42

mkdir -p "$OUT_DIR"

echo "========================================"
echo "Survey Defense Experiments - $TASK"
echo "Branch: experiment-1/implement-survey-defenses"
echo "Output: $OUT_DIR"
echo "========================================"

# Step 0: Clean accuracy baseline (includes new defenses)
echo "[Step 0] Evaluating clean accuracy with all defenses..."
python3 runs/eval_defense_accuracy.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR"

# Attackers to test (same as original-defenses for comparison + VIPER for unicode)
ATTACKERS=("BERTattack" "PWWS" "DeepWordBug" "Genetic" "VIPER")

for ATTACK in "${ATTACKERS[@]}"; do
    echo ""
    echo "========================================"
    echo "Attack: $ATTACK"
    echo "========================================"

    # Baseline (no defense)
    echo "[1/7] No defense..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense none

    # SpellCheck (comparison baseline)
    echo "[2/7] SpellCheck (baseline comparison)..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense spellcheck

    # Unicode Canonicalization
    echo "[3/7] Unicode Canonicalization..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense unicode

    # Majority Vote with 3, 5, 7 copies
    for COPIES in 3 5 7; do
        echo "[4-6/7] Majority Vote ($COPIES copies)..."
        python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
            "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
            --defense majority_vote --defense_param $COPIES --defense_seed $SEED
    done

    echo ""
    echo "Completed all defenses for $ATTACK"
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results saved to: $OUT_DIR"
echo "========================================"
echo ""
echo "Summary files to review:"
echo "  - Clean accuracy: $OUT_DIR/clean_accuracy_${TASK}_${VICTIM}.txt"
echo "  - Attack results: $OUT_DIR/results_*.txt"
