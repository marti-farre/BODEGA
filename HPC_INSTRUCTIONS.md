# HPC Instructions — UPF SNOW Cluster

This file documents how to run BODEGA experiments on the UPF HPC (SNOW) cluster.

## 1. Access

```bash
ssh username@hpc.s.upf.edu   # VPN required off-campus (UPF GlobalProtect)
```

## 2. One-time Environment Setup

Run this once in an interactive session:

```bash
# Request interactive node
salloc -p short --mem=16G --gres=gpu:1 -c 4

# Load conda
module load Miniconda3
eval "$(conda shell.bash hook)"

# Create environment
conda create -n bodega python=3.10 -y
conda activate bodega

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install "transformers==4.38.1" OpenAttack editdistance bert-score \
    symspellpy homoglyphs peft bitsandbytes accelerate

# For BLEURT (semantic scorer)
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git

# For Gemma (gated model) — needs HuggingFace account with Gemma access
huggingface-cli login   # enter your HF token when prompted
```

## 3. Upload Code and Data via SFTP

From your Mac:

```bash
sftp username@hpc.s.upf.edu

# Create directory
mkdir BODEGA

# Upload code (first time: clone from git instead)
# Then upload data:
put -r data/PR2 BODEGA/data/
put -r data/FC  BODEGA/data/
put -r data/HN  BODEGA/data/
put -r data/RD  BODEGA/data/
```

Alternatively, clone the git repo on HPC and only upload the data:
```bash
# On HPC
git clone <your-repo-url> ~/BODEGA
# Then upload just the data/ directory via SFTP
```

## 4. Prepare Working Directory on HPC

```bash
cd ~/BODEGA
mkdir -p logs results/experiment-7_multi
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
```

## 5. Train Victim Models (SLURM Array)

Submit training jobs for BERT and Gemma 2B across all 4 tasks in parallel:

```bash
sbatch scripts/slurm_train.sh
```

This submits 8 parallel jobs (array 0-7):
- Jobs 0-3: BERT on PR2, FC, HN, RD (~45 min each on T4)
- Jobs 4-7: Gemma 2B on PR2, FC, HN, RD (~3h each on T4)

Monitor:
```bash
squeue -u $USER
# Or check logs:
tail -f logs/train_<jobid>_<arrayid>.out
```

## 6. Run Attack Experiments (SLURM Arrays)

**Wait for all training jobs to complete before submitting attacks!**

```bash
# BERT attacks (32 jobs: 4 tasks × 4 attackers × 2 defenses)
# ~8h max (Genetic on T4)
sbatch scripts/slurm_attack_bert.sh

# Gemma attacks (32 jobs, same structure but larger GPU + more time)
sbatch scripts/slurm_attack_gemma.sh
```

You can submit both arrays at the same time; they run in parallel on different GPUs.

## 7. Download Results

Once all jobs finish, download results to your Mac:

```bash
sftp username@hpc.s.upf.edu
get -r BODEGA/results/experiment-7_multi ~/path/to/BODEGA/results/
```

Or use rsync for incremental sync:
```bash
rsync -avz username@hpc.s.upf.edu:~/BODEGA/results/experiment-7_multi/ \
    results/experiment-7_multi/
```

## 8. GPU Resource Notes

| GPU | VRAM | Best for |
|-----|------|---------|
| Tesla T4 | 16GB | BERT (training + inference), Gemma 2B (inference) |
| RTX 6000 | 24GB | Gemma 2B (faster), default for `slurm_attack_gemma.sh` |
| L40S | 48GB | Gemma 7B (if added later) |

The Gemma scripts request `--gres=gpu:rtx6000:1`. If no RTX 6000 is available:
- Change to `--gres=gpu:1` (any GPU, T4 likely) and increase `--time=14:00` to `24:00`
- Or use `--gres=gpu:l40s:1` if L40S nodes are free

## 9. Quota and Limits

- Max 5 GPUs simultaneously per user
- Short queue: 2h max | Medium queue: 8h max | High queue: 14 days
- The attack arrays (32 jobs) submit all at once; SLURM queues them — only 5 run simultaneously

## 10. Troubleshooting

**Job stuck in queue**: Check available resources with `sinfo` or cluster monitoring tool
**CUDA OOM**: Gemma 2B needs 8-10GB VRAM; T4 (16GB) should be sufficient for inference
**HuggingFace download fails**: Re-run `huggingface-cli login` in interactive session
**Module not found**: Make sure `eval "$(conda shell.bash hook)"` and `conda activate bodega` are in your job script
