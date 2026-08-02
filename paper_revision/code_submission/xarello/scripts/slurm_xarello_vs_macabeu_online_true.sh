#!/bin/bash
# SLURM array: XARELLO vs TRUE-ONLINE MACABEU (MV7 pseudo-label, soft reward)
#
# This is the deployment-realistic variant: MACABEU updates online using a
# MajorityVote-7 pseudo-label instead of the gold label. Soft reward =
# ±p where p is the winning vote fraction from MV7.
#
# 4 tasks × 3 victims = 12 jobs. Submit one victim at a time via env var:
#   sbatch scripts/slurm_xarello_vs_macabeu_online_true.sh
#   sbatch --export=ALL,XARELLO_VICTIM=BERT  scripts/slurm_xarello_vs_macabeu_online_true.sh
#   sbatch --export=ALL,XARELLO_VICTIM=GEMMA scripts/slurm_xarello_vs_macabeu_online_true.sh

#SBATCH -J xar_mac_true
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-3
#SBATCH -o logs/xar_mac_true_%A_%a.out
#SBATCH -e logs/xar_mac_true_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM="${XARELLO_VICTIM:-BiLSTM}"

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$i]}

DATA_PATH="$HOME/BODEGA/data/$TASK"
case "$VICTIM" in
    GEMMA) MODEL_PATH="$HOME/BODEGA/data/$TASK/GEMMA-512" ;;
    *)     MODEL_PATH="$HOME/BODEGA/data/$TASK/${VICTIM}-512.pth" ;;
esac
MACABEU_POLICY="$HOME/macabeu/models/${TASK}_${VICTIM}.pth"
OUT_DIR="results/xarello_vs_macabeu_online_true_soft/${VICTIM}"

CONDA_SH=/soft/easybuild/x86_64/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda.sh not found on $(hostname): $CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate bodega
export PYTHONPATH="$HOME/BODEGA:$PYTHONPATH"
export BODEGA_PATH="$HOME/BODEGA"
mkdir -p "$OUT_DIR" logs

echo "[$i] XARELLO vs TRUE-ONLINE MACABEU (mv7/soft) | $TASK | $VICTIM | warm=$MACABEU_POLICY"

python -m evaluation.attack \
    "$TASK" true XARELLO "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense macabeu_online --defense_policy "$MACABEU_POLICY" \
    --label_source mv7 --reward_mode soft \
    --semantic_scorer BLEURT
