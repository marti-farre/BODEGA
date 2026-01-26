#!/bin/bash
# =============================================================================
# Test preprocessing defenses against DeepWordBug on PR2 task
# =============================================================================
#
# This script runs 10 experiment configurations:
#   1. Baseline (no defense)
#   2. Spellcheck defense
#   3-6. Character dropout at 0.05, 0.10, 0.15, 0.20
#   7-10. Character noise at 0.05, 0.10, 0.15, 0.20
#
# Usage:
#   ./scripts/run_pr_defense_experiments.sh
#
# Prerequisites:
#   - Virtual environment activated
#   - Dependencies installed
#   - PR2 data converted and saved to data/PR2/
#   - BiLSTM model trained and saved to data/PR2/BiLSTM-512.pth
# =============================================================================

set -e  # Exit on error

# Set PYTHONPATH to find local modules
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

# Configuration
TASK="PR2"
TARGETED="false"
ATTACK="DeepWordBug"
VICTIM="BiLSTM"
DATA_PATH="data/PR2"
MODEL_PATH="data/PR2/BiLSTM-512.pth"
OUT_DIR="results"

# Create output directory
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Testing Defenses Against DeepWordBug (PR2)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Task: $TASK"
echo "  Targeted: $TARGETED"
echo "  Attack: $ATTACK"
echo "  Victim: $VICTIM"
echo "  Data: $DATA_PATH"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUT_DIR"
echo ""

# 1. Baseline (no defense)
echo "[1/10] Running baseline (no defense)..."
python runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense none
echo ""

# 2. Spellcheck defense
echo "[2/10] Running spellcheck defense..."
python runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense spellcheck
echo ""

# 3-6. Character Dropout at various rates
echo "[3/10] Running char_dropout at 0.05..."
python runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense char_dropout --defense_param 0.05 --defense_seed 42
echo ""

echo "[4/10] Running char_dropout at 0.10..."
python runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense char_dropout --defense_param 0.10 --defense_seed 42
echo ""

echo "[5/10] Running char_dropout at 0.15..."
python runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense char_dropout --defense_param 0.15 --defense_seed 42
echo ""

echo "[6/10] Running char_dropout at 0.20..."
python runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense char_dropout --defense_param 0.20 --defense_seed 42
echo ""

# 7-10. Character Noise at various rates
echo "[7/10] Running char_noise at 0.05..."
python runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense char_noise --defense_param 0.05 --defense_seed 42
echo ""

echo "[8/10] Running char_noise at 0.10..."
python runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense char_noise --defense_param 0.10 --defense_seed 42
echo ""

echo "[9/10] Running char_noise at 0.15..."
python runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense char_noise --defense_param 0.15 --defense_seed 42
echo ""

echo "[10/10] Running char_noise at 0.20..."
python runs/attack.py "$TASK" "$TARGETED" "$ATTACK" "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense char_noise --defense_param 0.20 --defense_seed 42
echo ""

echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUT_DIR"
echo ""
echo "Result files:"
ls -la "$OUT_DIR"/results_*.txt 2>/dev/null || echo "  (no result files found)"