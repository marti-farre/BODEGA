# Experiment 3: Combined SpellCheck + MajorityVote Defense

**Branch:** `experiment-3/combine-voting-check`
**Parent:** `main` (after merging experiment-1 and experiment-2)
**Status:** In progress
**Date:** March 2026

---

## Overview

This experiment evaluates a **combined defense** that chains SpellCheck and MajorityVote, aiming to cover both character-level and word-level attacks in a single wrapper.

### Motivation

From experiments 1 and 2:
- **SpellCheck** is the best defense against character-level attacks (DeepWordBug: -32pp ASR) but useless against word-level attacks
- **MajorityVote@7** (old) is the best defense against word-level attacks (BERTattack: -52pp ASR) but only moderately effective against character-level attacks
- Neither alone provides broad coverage

A combined SpellCheck→MajorityVote pipeline should address both attack types.

### Key Change from Experiment 1: MV Fix

The `MajorityVoteDefense` was updated in experiment-2:
- **Old**: `get_prob()` returned noisy aggregated probabilities (conditioned on noise)
- **New**: `get_prob()` returns clean victim probabilities; `get_pred()` uses majority voting

This change reduces MV's effectiveness against word-level attacks (see `results/majority_vote_fixed/SUMMARY_mv_fix_comparison.md`), as the clean probability signal helps attackers optimize more effectively. The combined defense aims to compensate.

---

## Summary of Changes vs Main

| File | Change |
|------|--------|
| `defenses/preprocessing.py` | Added `SpellCheckMVDefense` class + `'spellcheck_mv'` in `get_defense()` |
| `runs/eval_defense_accuracy.py` | Added `spellcheck_mv` configs (3, 5, 7 copies) |
| `scripts/run_experiment-3_combined-defenses.sh` | New experiment script |
| `explanations/experiment-3_combine-voting-check.md` | This file |

---

## Architecture

### Combined Defense Pipeline

```
Adversarial Input
       │
       ▼
┌──────────────────┐
│   SpellCheck     │  ← Corrects typos (char-level adversarial perturbations)
│   (SymSpell)     │    e.g. "happe" → "happy", "w0rk" → "work"
└──────┬───────────┘
       │ spellchecked text
       ▼
┌──────────────────────────────────┐
│         MajorityVote             │
│                                  │
│  for i in range(N):              │
│    perturbed = add_noise(text)   │  ← Random char ops (delete/swap/insert)
│    probs_i = victim(perturbed)   │
│                                  │
│  return majority_class(probs)    │  ← Robust aggregation
└──────────────────────────────────┘
       │
       ▼
   Prediction
```

### API Behavior

```
get_prob(adversarial_input):
    spellchecked = SpellCheck(adversarial_input)
    return victim.get_prob(spellchecked)          # clean, no noise

get_pred(adversarial_input):
    spellchecked = SpellCheck(adversarial_input)
    for N copies:
        noisy = add_mv_noise(spellchecked)
        probs.append(victim(noisy))
    return majority_vote(probs)
```

### Comparison with Individual Defenses

```
SpellCheck alone:
  adversarial → SpellCheck → victim → prediction

MajorityVote alone:
  adversarial → [N noisy copies] → victim × N → vote → prediction

SpellCheckMV (combined):
  adversarial → SpellCheck → [N noisy copies] → victim × N → vote → prediction
```

---

## Defense Implementation

**Class**: `SpellCheckMVDefense(MajorityVoteDefense)` in `defenses/preprocessing.py`

**Factory name**: `'spellcheck_mv'` (alias: `'sc_mv'`)

**Parameters**:
- `num_copies` (int): Number of MV perturbed copies (default 7; passed via `--defense_param`)
- `seed` (int): Random seed (passed via `--defense_seed`)

**Internal components**:
- `self._spellcheck`: `SpellCheckDefense` instance for text preprocessing
- Inherits `MajorityVoteDefense.defend_single()` for random character perturbations

---

## Experiment Design

### Defenses Compared

| Defense | Description | Expected Strength |
|---------|-------------|-------------------|
| none | Baseline | - |
| spellcheck | SpellCheck alone | Char-level attacks |
| majority_vote@7 | MV alone (fixed) | Word-level attacks (reduced vs old) |
| spellcheck_mv@7 | Combined | Both attack types |

### Fixed Attacker Benchmark

All experiments use the same 5 attackers for comparability across experiments:

| Attacker | Type | Notes |
|----------|------|-------|
| BERTattack | Word-level | Semantic substitutions |
| PWWS | Word-level | Saliency-weighted |
| DeepWordBug | Char-level | Typos (swaps, inserts, deletes) |
| Genetic | Word-level | Evolutionary |
| VIPER | Char-level | Visual homoglyphs |

---

## Usage

```bash
# Run all experiment-3 experiments
./scripts/run_experiment-3_combined-defenses.sh

# Run individual attack with combined defense
python3 runs/attack.py PR2 false DeepWordBug BiLSTM \
    data/PR2 data/PR2/BiLSTM-512.pth results/experiment-3_combine-voting-check \
    --defense spellcheck_mv --defense_param 7 --defense_seed 42

# Clean accuracy evaluation (includes spellcheck_mv)
python3 runs/eval_defense_accuracy.py PR2 BiLSTM data/PR2 data/PR2/BiLSTM-512.pth \
    results/experiment-3_combine-voting-check
```

---

## Expected Results

Based on individual defense results from experiments 1 and 2:

| Attack | Baseline | SpellCheck | MV@7 (fixed) | SpellCheck+MV@7 (expected) |
|--------|----------|------------|--------------|---------------------------|
| BERTattack | 82.5% | 82.2% | ~41% | ~35-40% |
| PWWS | 87.5% | 87.3% | ~58% | ~50-55% |
| DeepWordBug | 46.9% | 14.7% | ~37% | ~12-18% |
| Genetic | 89.7% | 87.7% | ~58% | ~50-55% |
| VIPER | 26.4% | 73.8% | ~29% | ~25-30% |

**Hypothesis**: SpellCheck+MV should perform at least as well as the better individual defense for each attack type:
- For DeepWordBug: dominated by SpellCheck (SpellCheck undoes typos before MV)
- For word-level: dominated by MV (SpellCheck has no effect on word substitutions)
- VIPER: uncertain (SpellCheck may hurt, MV may be neutral)

### Expected Clean Accuracy

SpellCheck has minimal accuracy cost (+0.4%), MV@7 costs -0.4%. Combined should be ~0% delta, or slightly negative due to occasional MV misclassification on spellchecked input.

---

## Results

*[To be filled after running experiments]*

See `results/experiment-3_combine-voting-check/SUMMARY_combined_defenses.md`

---

## Comparison Context

| Defense | Source | vs BERTattack | vs DeepWordBug | Utility Cost |
|---------|--------|---------------|----------------|--------------|
| SpellCheck | Exp 1 | -0.2pp | **-32.2pp** | +0.4% |
| MV@7 (old) | Exp 1 | **-52.2pp** | -20.7pp | -0.4% |
| MV@7 (fixed) | Exp 2 | -41.4pp | -9.9pp | -0.4% |
| LabelFlip@0.10 | Exp 2 | -44.7pp | -14.9pp | -4.7% |
| SpellCheck+MV@7 | **Exp 3** | TBD | TBD | TBD |

---

## Fixed Attacker Benchmark

Starting from experiment-3, all experiments use the **same 5 attackers** as a fixed evaluation suite:
- `BERTattack`, `PWWS`, `DeepWordBug`, `Genetic`, `VIPER`

This ensures comparability across experiments and prevents cherry-picking attackers.
