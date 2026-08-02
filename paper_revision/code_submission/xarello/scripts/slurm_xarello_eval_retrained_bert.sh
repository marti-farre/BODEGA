#!/bin/bash
# SLURM array job: Evaluate retrained XARELLO (BERT) on both no-defence and
# spellcheck_mv@3. Mirrors slurm_xarello_eval_retrained.sh (BiLSTM) so the
# BERT row of tab_xarello_adapt has the full pretrained/retrained x none/mv@3
# four-cell layout.
# 4 tasks x 2 eval defences = 8 jobs.
#
# Submit with:
#   cd ~/xarello
#   sbatch --dependency=afterok:<BERT_HN_TRAIN_JOBID> scripts/slurm_xarello_eval_retrained_bert.sh

#SBATCH -J xar_evr_b
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-7
#SBATCH -t 1-00:00:00
#SBATCH -o logs/xar_evr_b_%A_%a.out
#SBATCH -e logs/xar_evr_b_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM="BERT"
DEFENSE="spellcheck_mv"
DEFENSE_PARAM="3"

EVAL_DEFENSES=(
    none:0
    "${DEFENSE}:${DEFENSE_PARAM}"
)

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$((i / 2))]}
DEF_ENTRY=${EVAL_DEFENSES[$((i % 2))]}
EVAL_DEFENSE=${DEF_ENTRY%%:*}
EVAL_PARAM=${DEF_ENTRY##*:}

TRAINED_MODEL="models/trained_vs_${DEFENSE}/${TASK}_${VICTIM}/xarello-qmodel.pth"
DATA_PATH="$HOME/BODEGA/data/$TASK"
MODEL_PATH="$HOME/BODEGA/data/$TASK/${VICTIM}-512.pth"
OUT_DIR="results/xarello_retrained_vs_${DEFENSE}"

# Robust env activation (see slurm_xarello_train_bert_gemma.sh).
if [ -f /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh ]; then
    source /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
    conda activate bodega
else
    export PATH="$HOME/.conda/envs/bodega/bin:$PATH"
fi
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
export BODEGA_PATH="$HOME/BODEGA"
mkdir -p "$OUT_DIR" logs

echo "[$i] Retrained XARELLO | $TASK | $VICTIM | eval_defense=$EVAL_DEFENSE | trained_against=$DEFENSE"

if [ ! -f "$TRAINED_MODEL" ]; then
    echo "ERROR: Trained model not found at $TRAINED_MODEL"
    exit 1
fi

if [ "$EVAL_DEFENSE" = "none" ]; then
    python -m evaluation.attack \
        "$TASK" true XARELLO "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense none --qmodel_path "$TRAINED_MODEL" \
        --semantic_scorer BLEURT
else
    python -m evaluation.attack \
        "$TASK" true XARELLO "$VICTIM" \
        "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
        --defense "$EVAL_DEFENSE" --defense_param "$EVAL_PARAM" --defense_seed 42 \
        --qmodel_path "$TRAINED_MODEL" \
        --semantic_scorer BLEURT
fi
