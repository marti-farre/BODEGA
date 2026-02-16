#!/bin/bash
# Experiments for output perturbation defenses: Label Flipping, Random Threshold, Confidence Noise
# Branch: experiment-2/add-noise-to-output

set -e

# Set PYTHONPATH to find local modules
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

# Configuration
TASK="PR2"
VICTIM="BiLSTM"
DATA_PATH="data/PR2"
MODEL_PATH="data/PR2/BiLSTM-512.pth"
# Results go in branch-specific subdirectory
OUT_DIR="results/experiment-2_add-noise-to-output"
TARGETED="false"
SEED=42

mkdir -p "$OUT_DIR"

echo "========================================"
echo "Output Perturbation Defense Experiments - $TASK"
echo "Branch: experiment-2/add-noise-to-output"
echo "Output: $OUT_DIR"
echo "========================================"

# Step 0: Clean accuracy baseline (includes new defenses)
echo "[Step 0] Evaluating clean accuracy with all defenses..."
python3 runs/eval_defense_accuracy.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR"

# Attackers to test
ATTACKERS=("BERTattack" "PWWS" "DeepWordBug" "Genetic" "VIPER")

# Defense parameter values to test
PARAMS=(0.05 0.10 0.15)

for ATTACK in "${ATTACKERS[@]}"; do
    echo ""
    echo "========================================"
    echo "Attack: $ATTACK"
    echo "========================================"

    # Baseline (no defense)
    echo "[1/10] No defense..."
    python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense none

    # Exp 2a: Label Flipping
    STEP=2
    for PARAM in "${PARAMS[@]}"; do
        echo "[$STEP/10] Label Flipping (flip_prob=$PARAM)..."
        python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
            "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
            --defense label_flip --defense_param $PARAM --defense_seed $SEED
        ((STEP++))
    done

    # Exp 2b: Random Threshold
    for PARAM in "${PARAMS[@]}"; do
        echo "[$STEP/10] Random Threshold (range=$PARAM)..."
        python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
            "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
            --defense random_threshold --defense_param $PARAM --defense_seed $SEED
        ((STEP++))
    done

    # Exp 2c: Confidence Noise
    for PARAM in "${PARAMS[@]}"; do
        echo "[$STEP/10] Confidence Noise (std=$PARAM)..."
        python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
            "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
            --defense confidence_noise --defense_param $PARAM --defense_seed $SEED
        ((STEP++))
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
