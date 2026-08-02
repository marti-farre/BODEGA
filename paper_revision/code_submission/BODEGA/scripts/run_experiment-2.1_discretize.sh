#!/bin/bash
# Experiment 2.1: Discretized probability output defense
# Branch: experiment-2.1/discretize-probabilities
#
# Instead of returning soft probabilities [0.9, 0.1], the victim returns hard
# one-hot labels [1, 0] / [0, 1]. This removes ALL gradient information from
# the attacker's oracle while preserving clean accuracy exactly.
#
# No VIPER — VIPER is a char-level attacker that doesn't use probability gradients,
# so discretization provides no meaningful defense signal for it.

set -e

# Set PYTHONPATH to find local modules
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

# Configuration
TASK="PR2"
VICTIM="BiLSTM"
DATA_PATH="data/PR2"
MODEL_PATH="data/PR2/BiLSTM-512.pth"
OUT_DIR="results/experiment-2.1_discretize"
TARGETED="false"
SEED=42

mkdir -p "$OUT_DIR"

echo "========================================"
echo "Experiment 2.1: Discretized Probability Defense - $TASK"
echo "Branch: experiment-2.1/discretize-probabilities"
echo "Output: $OUT_DIR"
echo "========================================"

# Step 0: Clean accuracy baseline
echo "[Step 0] Evaluating clean accuracy..."
python3 runs/eval_defense_accuracy.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR"

# Attackers to test (no VIPER — char-level attack, unaffected by probability discretization)
ATTACKERS=("BERTattack" "PWWS" "DeepWordBug" "Genetic")

for ATTACK in "${ATTACKERS[@]}"; do
    echo ""
    echo "========================================"
    echo "Attack: $ATTACK"
    echo "========================================"

    # Baseline (no defense)
    echo "[1/2] No defense..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense none

    # Exp 2.1: Discretized probabilities
    echo "[2/2] Discretized probabilities..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense discretize --defense_seed $SEED

    echo ""
    echo "Completed for $ATTACK"
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
