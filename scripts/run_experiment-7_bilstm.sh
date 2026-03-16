#!/bin/bash
# Experiment 7: Multi-task × Multi-victim — BiLSTM portion (runs on Mac)
# Branch: experiment-7/multi-task-multi-victim
#
# Runs ALL defenses from experiments 1-4 across all 4 tasks (PR2, FC, HN, RD).
# Matches the full defense suite that was run for PR2/BiLSTM in experiments 1-4.
#
# Defense count: 28 conditions × 4 attackers × 4 tasks = 448 experiments
# Estimated time: ~3-4 hours on Mac M2

set -e

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

VICTIM="BiLSTM"
OUT_DIR="results/experiment-7_multi"
TARGETED="false"
SEED=42

mkdir -p "$OUT_DIR"

echo "========================================"
echo "Experiment 7: Multi-task BiLSTM — PR2, FC, HN, RD (all defenses)"
echo "Branch: experiment-7/multi-task-multi-victim"
echo "Output: $OUT_DIR"
echo "========================================"

ATTACKERS=("BERTattack" "PWWS" "DeepWordBug" "Genetic")
TASKS=("PR2" "FC" "HN" "RD")

for TASK in "${TASKS[@]}"; do
    DATA_PATH="data/$TASK"
    MODEL_PATH="data/$TASK/BiLSTM-512.pth"

    echo ""
    echo "========================================"
    echo "Task: $TASK (BiLSTM)"
    echo "========================================"

    # Clean accuracy (all defenses)
    echo "[Step 0] Clean accuracy for $TASK..."
    python3 runs/eval_defense_accuracy.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR"

    for ATTACK in "${ATTACKERS[@]}"; do
        echo ""
        echo "--- $TASK / $ATTACK ---"

        # Baseline
        python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
            "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" --defense none

        # Exp 1: Input defenses
        python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
            "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" --defense spellcheck --defense_seed $SEED

        for PARAM in 0.05 0.10 0.15 0.20; do
            python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense char_noise --defense_param $PARAM --defense_seed $SEED
            python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense char_masking --defense_param $PARAM --defense_seed $SEED
        done

        python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
            "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" --defense unicode --defense_seed $SEED

        for COPIES in 3 5 7; do
            python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense majority_vote --defense_param $COPIES --defense_seed $SEED
        done

        # Exp 2: Output defenses
        for PARAM in 0.05 0.10 0.15; do
            python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense label_flip --defense_param $PARAM --defense_seed $SEED
            python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense random_threshold --defense_param $PARAM --defense_seed $SEED
            python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense confidence_noise --defense_param $PARAM --defense_seed $SEED
        done

        # Exp 2.1: Discretized probabilities
        python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
            "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" --defense discretize --defense_seed $SEED

        # Exp 3: SpellCheck + MajorityVote
        for COPIES in 3 7; do
            python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense spellcheck_mv --defense_param $COPIES --defense_seed $SEED
        done

        # Exp 4: Unicode + MajorityVote
        for COPIES in 3 7; do
            python3 runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
                "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
                --defense unicode_mv --defense_param $COPIES --defense_seed $SEED
        done
    done
done

echo ""
echo "========================================"
echo "BiLSTM experiments completed!"
echo "Results: $OUT_DIR"
echo "========================================"
