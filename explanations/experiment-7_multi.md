# Branch: experiment-7/multi-task-multi-victim

## Overview

**Purpose**: Validate that the SC+MV@3 defense generalizes across NLP tasks and victim models beyond PR2/BiLSTM.

**Parent Branch**: `main` (merged from `experiment-2.1/discretize-probabilities`)

**Status**: In progress

---

## Experiment Matrix

| | BiLSTM | BERT | Gemma 2B |
|---|---|---|---|
| **PR2** | ✅ done (Exp 3/5) | HPC | HPC |
| **FC** | Mac | HPC | HPC |
| **HN** | Mac | HPC | HPC |
| **RD** | Mac | HPC | HPC |

- **Attackers**: BERTattack, PWWS, DeepWordBug, Genetic (no VIPER)
- **Defenses**: none (baseline), spellcheck_mv@3
- **Total**: 4 tasks × 3 victims × 4 attackers × 2 conditions = 96 experiments

---

## Tasks

| ID | Name | Dataset | Pairs? | Notes |
|----|------|---------|--------|-------|
| PR2 | Propaganda Detection | SemEval-2020 Task 11 | No | Single article sentences |
| FC | Fact Checking | FEVER | Yes (claim ~ evidence) | Evidence from Wikipedia |
| HN | Hyperpartisan News | SemEval-2019 Task 4 | No | Full news articles |
| RD | Rumour Detection | Aug-RNR (PHEME) | No | Tweet threads |

## Victim Models

| Model | Architecture | Training | Output format |
|-------|-------------|----------|---------------|
| BiLSTM | BiLSTM-128 + Linear | 10 epochs, lr=1e-3 | .pth state dict |
| BERT | bert-base-uncased + head | 5 epochs, lr=5e-5 | .pth state dict |
| Gemma 2B | google/gemma-2b + QLoRA | 5 epochs, lr=5e-5 | directory (PEFT) |

---

## Files Changed

| File | Change |
|------|--------|
| `scripts/run_experiment-7_bilstm.sh` | BiLSTM attack experiments (Mac) |
| `scripts/slurm_train.sh` | SLURM array: train BERT + Gemma for all tasks |
| `scripts/slurm_attack_bert.sh` | SLURM array: 32 BERT attack runs |
| `scripts/slurm_attack_gemma.sh` | SLURM array: 32 Gemma attack runs |
| `explanations/experiment-7_multi.md` | This file |

---

## Running on Mac (BiLSTM)

```bash
# Step 1: Convert datasets (one-time, if not done)
python conversion/convert_FC.py ~/Downloads/bodega-datasets/fever data/FC
python conversion/convert_HN.py ~/Downloads/bodega-datasets/hn data/HN
python conversion/convert_RD.py ~/Downloads/bodega-datasets/rd/aug-rnr-data_filtered data/RD

# Step 2: Train BiLSTM for each task
for TASK in FC HN RD; do
  python runs/train_victims.py $TASK BiLSTM data/$TASK data/$TASK/BiLSTM-512.pth
done

# Step 3: Run attack experiments
./scripts/run_experiment-7_bilstm.sh
```

---

## Running on HPC (BERT + Gemma)

See `HPC_INSTRUCTIONS.md` at the repo root for full setup guide.

```bash
# Submit training array (8 jobs: BERT×4tasks + GEMMA×4tasks)
mkdir -p logs
sbatch scripts/slurm_train.sh

# Monitor
squeue -u $USER

# After all training jobs finish, submit attack arrays
sbatch scripts/slurm_attack_bert.sh    # 32 BERT runs
sbatch scripts/slurm_attack_gemma.sh   # 32 Gemma runs (large GPU)
```

---

## Result Files

```
results/experiment-7_multi/
├── clean_accuracy_<task>_<victim>.txt     (one per task/victim)
├── results_<task>_False_<attack>_<victim>.txt         (baseline)
├── results_<task>_False_<attack>_<victim>_spellcheck_mv_3.0.txt  (SC+MV@3)
└── SUMMARY_multi.md                       (filled after all results collected)
```
