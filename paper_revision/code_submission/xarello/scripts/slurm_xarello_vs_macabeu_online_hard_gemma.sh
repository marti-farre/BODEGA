#!/bin/bash
# SLURM array: XARELLO vs TRUE-ONLINE MACABEU (mv7/hard) on GEMMA.
# Fills the GEMMA hole in the hard grid so Table 1's MACABEU-est row covers
# the full 12 sub-combos including XARELLO. 4 cells = 4 tasks x GEMMA.
#
# Submit:
#   sbatch --exclude=node023 scripts/slurm_xarello_vs_macabeu_online_hard_gemma.sh

#SBATCH -J xar_mac_hard_g
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-3
#SBATCH -o logs/xar_mac_hard_g_%A_%a.out
#SBATCH -e logs/xar_mac_hard_g_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIM=GEMMA

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$i]}

DATA_PATH="$HOME/BODEGA/data/$TASK"
MODEL_PATH="$HOME/BODEGA/data/$TASK/GEMMA-512"
MACABEU_POLICY="$HOME/macabeu/models/${TASK}_${VICTIM}.pth"
OUT_DIR="results/xarello_vs_macabeu_online_true_hard/${VICTIM}"

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

echo "[$i] XARELLO vs TRUE-ONLINE MACABEU (mv7/hard) GEMMA-fill | $TASK | $VICTIM | warm=$MACABEU_POLICY"

python -m evaluation.attack \
    "$TASK" true XARELLO "$VICTIM" \
    "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --defense macabeu_online --defense_policy "$MACABEU_POLICY" \
    --label_source mv7 --reward_mode hard \
    --semantic_scorer BLEURT
