# BODEGA - Development Guide

This document provides comprehensive documentation for working with the BODEGA codebase, particularly for the thesis work on preprocessing defenses against adversarial attacks on misinformation detection.

---

## 1. Project Overview

**BODEGA** (Benchmark for Adversarial Example Generation in Credibility Assessment) is a benchmark for evaluating the robustness of text classifiers against adversarial attacks. It focuses on misinformation detection tasks and simulates real-world scenarios where ML classifiers are used for content filtering on social media platforms.

### Thesis Focus
This fork extends BODEGA with **preprocessing defenses** - input transformations applied at inference time to counter adversarial attacks, particularly character-level attacks like DeepWordBug.

### Related Publication
- Original BODEGA: [Verifying the Robustness of Automatic Credibility Assessment](https://doi.org/10.1017/nlp.2024.54) (NLP Journal)
- XARELLO: [Know Thine Enemy: Adaptive Attacks on Misinformation Detection Using Reinforcement Learning](https://aclanthology.org/2024.wassa-1.11/) (WASSA @ ACL 2024)

---

## 2. Directory Structure

```
BODEGA/
├── CLAUDE.md              # This file - development guide
├── README.MD              # Original BODEGA readme
├── conversion/            # Scripts to convert source corpora to BODEGA format
│   ├── convert_PR2.py     # Propaganda detection
│   ├── convert_FC.py      # Fact checking
│   ├── convert_HN.py      # Hyperpartisan news
│   ├── convert_RD.py      # Rumour detection
│   └── convert_C19.py     # COVID-19 misinformation
├── data/                  # Data directory (not in git)
│   └── PR2/               # Task-specific data
│       ├── train.tsv
│       ├── attack.tsv
│       ├── dev.tsv
│       └── BiLSTM-512.pth # Trained model
├── defenses/              # Defense implementations
│   ├── __init__.py
│   └── preprocessing.py   # SpellCheck, CharNoise, CharMasking defenses
├── explanations/          # Branch-level documentation
│   └── experiment-1_add-noise-to-input.md
├── metrics/               # Evaluation metrics
│   └── BODEGAScore.py     # Combined success/semantic/character score
├── results/               # Experiment results
│   └── SUMMARY_preprocessing_defenses.md
├── runs/                  # Main execution scripts
│   ├── attack.py          # Run attacks with optional defenses
│   ├── eval_defense_accuracy.py  # Evaluate clean accuracy
│   └── train_victims.py   # Train victim classifiers
├── scripts/               # Automation scripts
│   ├── run_pr_defense_experiments.sh
│   └── run_multi_attacker_experiments.sh
├── utils/                 # Utility functions
│   ├── data_mappings.py   # Dataset format conversions
│   └── no_ssl_verify.py   # SSL workaround for OpenAttack
└── victims/               # Victim model implementations
    ├── bilstm.py          # BiLSTM classifier
    ├── transformer.py     # BERT/Gemma classifiers
    ├── surprise.py        # RoBERTa surprise classifier
    └── caching.py         # Prediction caching wrapper
```

---

## 3. Key Files

### Defense Implementation
- **[defenses/preprocessing.py](defenses/preprocessing.py)** - All defense classes:
  - `DefenseWrapper` - Base class that wraps OpenAttack classifiers
  - `SpellCheckDefense` - Corrects misspellings using SymSpell
  - `CharacterNoiseDefense` - Substitutes chars with Unicode homoglyphs
  - `CharacterMaskingDefense` - Randomly removes characters
  - `IdentityDefense` - No-op baseline
  - `get_defense()` - Factory function to create defenses

### Attack Evaluation
- **[runs/attack.py](runs/attack.py)** - Main attack script with defense integration
- **[runs/eval_defense_accuracy.py](runs/eval_defense_accuracy.py)** - Clean accuracy evaluation

### Training
- **[runs/train_victims.py](runs/train_victims.py)** - Train BiLSTM/BERT/Gemma classifiers

### Metrics
- **[metrics/BODEGAScore.py](metrics/BODEGAScore.py)** - Computes:
  - Success score (attack success rate)
  - Semantic score (BERTScore/BLEURT similarity)
  - Character score (Levenshtein distance)
  - BODEGA score (combined metric)

---

## 4. Available Defenses

| Defense | Parameter | Description |
|---------|-----------|-------------|
| `none` | - | No defense (baseline) |
| `spellcheck` | - | Corrects spelling errors using SymSpell dictionary |
| `char_noise` | `noise_std` (0.0-1.0) | Substitutes characters with Unicode homoglyphs |
| `char_masking` | `masking_prob` (0.0-1.0) | Randomly removes characters |
| `identity` | - | No-op (useful for testing) |

### Defense Parameters
- `char_noise`: Typical values are 0.05, 0.10, 0.15, 0.20
- `char_masking`: Typical values are 0.05, 0.10, 0.15, 0.20

### Dependencies
```bash
pip install symspellpy   # For SpellCheckDefense
pip install homoglyphs   # For CharacterNoiseDefense
```

---

## 5. Available Attackers

All attackers are from the OpenAttack library:

| Attacker | Type | Description |
|----------|------|-------------|
| `DeepWordBug` | Character-level | Introduces typos (swaps, insertions, deletions) |
| `TextFooler` | Word-level | Replaces words with synonyms |
| `BERTattack` | Word-level | Uses BERT to find word replacements |
| `PWWS` | Word-level | Probability Weighted Word Saliency |
| `Genetic` | Word-level | Genetic algorithm-based attack |
| `PSO` | Word-level | Particle Swarm Optimization attack |
| `BAE` | Word-level | BERT-based Adversarial Examples |
| `SCPN` | Sentence-level | Syntactic paraphrase attack |
| `GAN` | Word-level | GAN-based attack |
| `VIPER` | Character-level | Visual perturbations |

---

## 6. Common Commands

### Run Attack with Defense
```bash
python runs/attack.py <task> <targeted> <attack> <victim> <data_path> <model_path> <output_dir> \
    --defense <defense_type> --defense_param <param> --defense_seed <seed>

# Example: DeepWordBug attack on PR2 with spellcheck defense
python runs/attack.py PR2 false DeepWordBug BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results \
    --defense spellcheck

# Example: BERTattack with character noise defense at 10%
python runs/attack.py PR2 false BERTattack BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results \
    --defense char_noise --defense_param 0.10 --defense_seed 42
```

### Evaluate Clean Accuracy
```bash
python runs/eval_defense_accuracy.py <task> <victim> <data_path> <model_path> [output_dir]

# Example
python runs/eval_defense_accuracy.py PR2 BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results
```

### Train Victim Model
```bash
python runs/train_victims.py <task> <model_type> <data_path> <output_path>

# Example
python runs/train_victims.py PR2 BiLSTM data/PR2 data/PR2/BiLSTM-512.pth
```

### Run All Defense Experiments
```bash
./scripts/run_pr_defense_experiments.sh
./scripts/run_multi_attacker_experiments.sh
```

---

## 7. Data Format

### TSV File Structure
All data files are TSV (tab-separated) with three columns:
1. **Label**: 1 (non-credible/propaganda/fake) or 0 (credible/real)
2. **Source**: URL or identifier
3. **Text**: Content text (newlines encoded as `\n`)

### Task IDs
| ID | Task | Description |
|----|------|-------------|
| `PR2` | Propaganda Detection | SemEval-2020 Task 11 |
| `FC` | Fact Checking | FEVER dataset |
| `HN` | Hyperpartisan News | PAN @ SemEval 2019 |
| `RD` | Rumour Detection | Augmented RNR dataset |
| `C19` | COVID-19 Misinfo | CheckThat! 2024 |

### Text Pairs
For FC and C19 tasks, text pairs are joined with separator ` ~ `:
```
claim text ~ evidence text
```

---

## 8. XARELLO Reference

**XARELLO** (eXploring Adversarial examples using REinforcement Learning Optimisation) is a sibling project that implements adaptive adversarial attacks using reinforcement learning.

### Location
```
../xarello/   # Sibling directory to BODEGA
```

### Key Differences from Standard Attacks
- **Adaptive**: Learns which attacks work against specific classifiers
- **RL-based**: Uses Q-learning with Transformer-based Q estimator
- **Targeted**: Optimizes for successful label flips

### Integration with BODEGA
- XARELLO requires BODEGA in PYTHONPATH
- Uses same data format and victim models
- Results can be compared using BODEGA metrics

### Key Files in XARELLO
- `main-train-eval.py` - Train XARELLO agent
- `evaluation/attack.py` - Evaluate XARELLO attack
- `agent/` - RL agent implementation
- `defenses/` - Original defense implementations (adapted for BODEGA)

---

## 9. Explanation Directory

The `explanations/` directory documents changes at the branch level, providing:
- Summary of modifications
- Architecture diagrams
- Comparison with main branch
- Results highlights

### Naming Convention
```
explanations/<branch-name>.md
```

Example: `explanations/experiment-1_add-noise-to-input.md`

### Structure
Each explanation file contains:
1. Branch overview and purpose
2. Summary of changes vs main
3. Architecture/flow diagrams
4. Files changed table
5. Key results and findings

---

## 10. Current Experiment Status

### Active Branch
`experiment-1/add-noise-to-input`

### Implemented Features
- Preprocessing defense framework
- SpellCheck defense (SymSpell-based)
- Character Noise defense (homoglyphs-based)
- Character Masking defense
- Clean accuracy evaluation script
- Multi-attacker experiment script

### Results Location
- Summary: `results/SUMMARY_preprocessing_defenses.md`
- Raw results: `results/results_*.txt`
- Clean accuracy: `results/clean_accuracy_*.txt`

### Key Findings (DeepWordBug on PR2)
| Defense | Accuracy Delta | ASR Reduction |
|---------|---------------|---------------|
| SpellCheck | +0.4% | -32.2pp |
| char_masking@0.10 | -6.5% | -26.9pp |
| char_noise@0.10 | -4.7% | -20.7pp |

---

## 11. Development Workflow

### Adding a New Defense
1. Create class inheriting from `DefenseWrapper` in `defenses/preprocessing.py`
2. Implement `defend_single(self, text: str) -> str`
3. Add to `get_defense()` factory function
4. Add to `DEFENSE_CONFIGS` in `eval_defense_accuracy.py`
5. Test with `--verbose` flag to see modifications

### Running Experiments
1. Ensure virtual environment is activated
2. Run clean accuracy evaluation first
3. Run attack experiments with various defenses
4. Collect results and update summary

### Debugging
```python
# Test defense in isolation
from defenses.preprocessing import get_defense

class MockVictim:
    def get_prob(self, texts):
        import numpy as np
        return np.array([[0.3, 0.7]] * len(texts))

victim = MockVictim()
defended = get_defense('char_noise', victim, param=0.1, seed=42, verbose=True)
defended.get_pred(["This is a test sentence."])
```

---

## 12. Environment Setup

### Requirements
```bash
# Core
conda create -n bodega python=3.10
conda activate bodega
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# NLP
pip install "transformers==4.38.1"
pip install OpenAttack

# Metrics
pip install editdistance bert-score
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
pip install git+https://gitlab.clarin-pl.eu/syntactic-tools/lambo.git

# Defenses
pip install symspellpy homoglyphs

# Gemma models (optional)
pip install peft bitsandbytes accelerate
```

### PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
```

---

## 13. Troubleshooting

### SSL Certificate Errors
OpenAttack may fail due to outdated SSL certificates. The codebase includes `utils/no_ssl_verify.py` to work around this.

### CUDA Memory Issues
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:100`
- For TextFooler/BAE, TensorFlow GPU is disabled automatically

### Missing Dependencies
```bash
pip install symspellpy   # SpellCheckDefense
pip install homoglyphs   # CharacterNoiseDefense
```

### Slow Evaluation
- Use `VictimCache` wrapper for repeated predictions
- Defense wrappers disable caching (input transforms make it invalid)
