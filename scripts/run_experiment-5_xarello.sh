#!/bin/bash
# Experiment 5: XARELLO Adaptive Attacker vs SC+MV@7 Defense
#
# Two sub-experiments:
#   5.1 Pre-trained XARELLO: train on undefended victim, evaluate against defended victim
#   5.2 Adaptive XARELLO:    train on defended victim, evaluate against defended victim
#
# Both train on PR2/BiLSTM (same as previous experiments).
# Defense: spellcheck_mv@7 (best broad coverage from experiment-3).
#
# Prerequisites:
#   - XARELLO repo at ../xarello/ on branch experiment-5/bodega-defenses
#   - Data at ~/data/BODEGA/PR2/ and ~/data/xarello/
#   - PYTHONPATH includes BODEGA root
#
# Branch: experiment-5/xarello (BODEGA), experiment-5/bodega-defenses (XARELLO)

set -e

# Resolve absolute paths before any cd (relative paths break after cd)
BODEGA_ABS="$(cd "$(dirname "$0")/.." && pwd)"
XARELLO_ABS="$(cd "${BODEGA_ABS}/../xarello" && pwd)"

TASK="PR2"
VICTIM="BiLSTM"
DEFENSE="spellcheck_mv"
DEFENSE_PARAM="7"
DEFENSE_SEED="42"

MODEL_DIR_51="${HOME}/data/xarello/models/exp5.1/${TASK}-${VICTIM}"
MODEL_DIR_52="${HOME}/data/xarello/models/exp5.2/${TASK}-${VICTIM}"
OUT_DIR_51="${BODEGA_ABS}/results/experiment-5_xarello/5.1-pretrained"
OUT_DIR_52="${BODEGA_ABS}/results/experiment-5_xarello/5.2-adaptive"

mkdir -p "$MODEL_DIR_51" "$MODEL_DIR_52" "$OUT_DIR_51" "$OUT_DIR_52"

# XARELLO hardcodes ~/data/BODEGA/<TASK>/ for model and data loading.
# Create a symlink if the expected path doesn't already exist.
if [ ! -e "${HOME}/data/BODEGA" ]; then
    mkdir -p "${HOME}/data"
    ln -s "${BODEGA_ABS}/data" "${HOME}/data/BODEGA"
    echo "Created symlink: ~/data/BODEGA -> ${BODEGA_ABS}/data"
fi

# Both BODEGA and XARELLO must be on PYTHONPATH as absolute paths
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${BODEGA_ABS}:${XARELLO_ABS}"

echo "========================================================"
echo "Experiment 5: XARELLO Adaptive Attacker vs SC+MV@7"
echo "Task: $TASK, Victim: $VICTIM, Defense: ${DEFENSE}@${DEFENSE_PARAM}"
echo "========================================================"

# ─────────────────────────────────────────────────────────────
# EXPERIMENT 5.1 — Pre-trained XARELLO vs defense at inference
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== 5.1: Train XARELLO on undefended victim ==="
echo "Output: $MODEL_DIR_51"

(cd "$XARELLO_ABS" && \
  python main-train-eval.py "$TASK" "$VICTIM" "$MODEL_DIR_51")

echo ""
echo "=== 5.1: Evaluate pre-trained XARELLO — no defense (baseline) ==="
(cd "$XARELLO_ABS" && \
  python evaluation/attack.py \
    --task "$TASK" --victim "$VICTIM" \
    --qmodel_path "${MODEL_DIR_51}/xarello-qmodel.pth" \
    --out_dir "$OUT_DIR_51" \
    --defense none)

echo ""
echo "=== 5.1: Evaluate pre-trained XARELLO — SC+MV@7 defense ==="
(cd "$XARELLO_ABS" && \
  python evaluation/attack.py \
    --task "$TASK" --victim "$VICTIM" \
    --qmodel_path "${MODEL_DIR_51}/xarello-qmodel.pth" \
    --out_dir "$OUT_DIR_51" \
    --defense "$DEFENSE" --defense_param "$DEFENSE_PARAM" --defense_seed "$DEFENSE_SEED")

# ─────────────────────────────────────────────────────────────
# EXPERIMENT 5.2 — Adaptive XARELLO: retrain with defense in loop
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== 5.2: Train XARELLO on defended victim (${DEFENSE}@${DEFENSE_PARAM}) ==="
echo "Output: $MODEL_DIR_52"

(cd "$XARELLO_ABS" && \
  python main-train-eval.py "$TASK" "$VICTIM" "$MODEL_DIR_52" \
    none 0.0 42 \
    "$DEFENSE" "$DEFENSE_PARAM" "$DEFENSE_SEED")

echo ""
echo "=== 5.2: Evaluate adaptive XARELLO — no defense (baseline) ==="
(cd "$XARELLO_ABS" && \
  python evaluation/attack.py \
    --task "$TASK" --victim "$VICTIM" \
    --qmodel_path "${MODEL_DIR_52}/xarello-qmodel.pth" \
    --out_dir "$OUT_DIR_52" \
    --defense none)

echo ""
echo "=== 5.2: Evaluate adaptive XARELLO — SC+MV@7 defense ==="
(cd "$XARELLO_ABS" && \
  python evaluation/attack.py \
    --task "$TASK" --victim "$VICTIM" \
    --qmodel_path "${MODEL_DIR_52}/xarello-qmodel.pth" \
    --out_dir "$OUT_DIR_52" \
    --defense "$DEFENSE" --defense_param "$DEFENSE_PARAM" --defense_seed "$DEFENSE_SEED")

echo ""
echo "========================================================"
echo "Experiment 5 complete."
echo "  5.1 results: $OUT_DIR_51"
echo "  5.2 results: $OUT_DIR_52"
echo "========================================================"
