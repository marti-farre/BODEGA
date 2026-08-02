#!/bin/bash
# Experiment 3: Combined SpellCheck + MajorityVote defense evaluation.
# Compares individual defenses (SpellCheck, MV@7) against the combined (SpellCheckMV@7).
# Branch: experiment-3/combine-voting-check

set -e

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

TASK="PR2"
VICTIM="BiLSTM"
DATA_PATH="data/PR2"
MODEL_PATH="data/PR2/BiLSTM-512.pth"
OUT_DIR="results/experiment-3_combine-voting-check"
TARGETED="false"
SEED=42

mkdir -p "$OUT_DIR"

echo "========================================"
echo "Experiment 3: Combined Defense (SpellCheck + MajorityVote)"
echo "Branch: experiment-3/combine-voting-check"
echo "Output: $OUT_DIR"
echo "========================================"

# Fixed attacker benchmark (same 5 across all experiments)
ATTACKERS=("BERTattack" "PWWS" "DeepWordBug" "Genetic" "VIPER")

# Step 0: Clean accuracy evaluation (includes spellcheck_mv)
echo "[Step 0] Evaluating clean accuracy with all defenses..."
python3 runs/eval_defense_accuracy.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR"

for ATTACK in "${ATTACKERS[@]}"; do
    echo ""
    echo "========================================"
    echo "Attack: $ATTACK"
    echo "========================================"

    # 1. Baseline (no defense)
    echo "[1/4] No defense..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense none

    # 2. SpellCheck alone
    echo "[2/4] SpellCheck..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense spellcheck --defense_seed $SEED

    # 3. MajorityVote@7 alone (fixed version)
    echo "[3/4] MajorityVote@7 (fixed)..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense majority_vote --defense_param 7 --defense_seed $SEED

    # 4. Combined: SpellCheck + MajorityVote@7
    echo "[4/4] SpellCheck + MajorityVote@7 (combined)..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense spellcheck_mv --defense_param 7 --defense_seed $SEED

    echo "Completed $ATTACK"
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
