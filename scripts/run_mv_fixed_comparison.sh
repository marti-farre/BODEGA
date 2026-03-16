#!/bin/bash
# Re-run Majority Vote experiments with the fixed implementation.
# Fixed MV: get_prob() returns clean probabilities, get_pred() uses voting.
# Compare results against experiment-1_implement-survey-defenses baseline.
# Branch: experiment-2/add-noise-to-output

set -e

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

TASK="PR2"
VICTIM="BiLSTM"
DATA_PATH="data/PR2"
MODEL_PATH="data/PR2/BiLSTM-512.pth"
OUT_DIR="results/majority_vote_fixed"
TARGETED="false"
SEED=42

mkdir -p "$OUT_DIR"

echo "========================================"
echo "Majority Vote (FIXED) - Comparison Experiments"
echo "Branch: experiment-2/add-noise-to-output"
echo "Output: $OUT_DIR"
echo "========================================"
echo "NOTE: Comparing against results/experiment-1_implement-survey-defenses/"
echo ""

# Clean accuracy with fixed MV
echo "[Step 0] Evaluating clean accuracy (fixed MV)..."
python3 runs/eval_defense_accuracy.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR"

ATTACKERS=("BERTattack" "PWWS" "DeepWordBug" "Genetic" "VIPER")
COPIES=(3 5 7)

for ATTACK in "${ATTACKERS[@]}"; do
    echo ""
    echo "========================================"
    echo "Attack: $ATTACK"
    echo "========================================"

    # Baseline (no defense)
    echo "[1] No defense baseline..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense none

    # Fixed Majority Vote with 3, 5, 7 copies
    STEP=2
    for N in "${COPIES[@]}"; do
        echo "[$STEP] Majority Vote (fixed, num_copies=$N)..."
        python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
            "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
            --defense majority_vote --defense_param $N --defense_seed $SEED
        ((STEP++))
    done

    echo "Completed $ATTACK"
done

echo ""
echo "========================================"
echo "All fixed-MV experiments completed!"
echo "Results saved to: $OUT_DIR"
echo "Compare against: results/experiment-1_implement-survey-defenses/"
echo "========================================"
