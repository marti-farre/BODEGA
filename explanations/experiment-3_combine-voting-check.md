# Experiment 3: Combined SpellCheck + MajorityVote Defense

**Branch:** `experiment-3/combine-voting-check`
**Parent:** `main` (after merging experiment-1 and experiment-2)
**Status:** Completed
**Date:** March 2026

---

## Overview

This experiment evaluates a **combined defense** that chains SpellCheck and MajorityVote, aiming to cover both character-level and word-level attacks in a single wrapper.

### Motivation

From experiments 1 and 2:
- **SpellCheck** is the best defense against character-level attacks (DeepWordBug: -32pp ASR) but useless against word-level attacks
- **MajorityVote@7** is the best defense against word-level attacks (BERTattack: -52pp ASR) but only moderately effective against character-level attacks
- Neither alone provides broad coverage

A combined SpellCheck→MajorityVote pipeline should address both attack types.

### Note on MajorityVote Implementation

During this experiment, we explored whether `get_prob()` in `MajorityVoteDefense` should return clean victim probabilities (decoupled from noise) or the original noisy vote-fraction probabilities. We initially implemented the "clean" version, compared results, and **reverted to the original noisy oracle**:

- **Original (retained)**: `get_prob()` returns vote-fraction probabilities from N noisy copies — a stochastic oracle that confuses gradient-based attackers. Aligned with randomized smoothing theory (Cohen et al. 2019).
- **"Fixed" (discarded)**: `get_prob()` returns clean victim probabilities; `get_pred()` votes separately. This made MV 10-21pp worse for word-level attacks (BERTattack: 30.3% → 41.1% ASR; PWWS: 36.3% → 57.7% ASR).

The comparison is documented in `results/majority_vote_fixed/SUMMARY_mv_fix_comparison.md`.

**Conclusion**: The noisy oracle is a feature, not a bug — it is the core mechanism by which MV disrupts word-level adversarial search.

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
│  return vote_fractions(probs)    │  ← Noisy oracle (stochastic probabilities)
└──────────────────────────────────┘
       │
       ▼
   Prediction
```

### API Behavior

```
get_prob(adversarial_input):
    spellchecked = SpellCheck(adversarial_input)
    # Run N noisy copies → return vote-fraction probabilities (noisy oracle)
    for N copies:
        noisy = add_mv_noise(spellchecked)
        probs.append(victim(noisy))
    return vote_fractions(probs)

get_pred(adversarial_input):
    return get_prob(adversarial_input).argmax()   # majority class
```

### Comparison with Individual Defenses

```
SpellCheck alone:
  adversarial → SpellCheck → victim → prediction

MajorityVote alone:
  adversarial → [N noisy copies] → victim × N → vote fractions → argmax

SpellCheckMV (combined):
  adversarial → SpellCheck → [N noisy copies] → victim × N → vote fractions → argmax
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
- Inherits `MajorityVoteDefense.get_prob()` for noisy oracle (N copies → vote fractions)
- Inherits `MajorityVoteDefense.defend_single()` for random character perturbations

---

## Experiment Design

### Defenses Compared

| Defense | Description | Expected Strength |
|---------|-------------|-------------------|
| none | Baseline | - |
| spellcheck | SpellCheck alone | Char-level attacks |
| majority_vote@7 | MV alone (noisy oracle) | Word-level attacks |
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

## Results

See `results/experiment-3_combine-voting-check/SUMMARY_combined_defenses.md` for the full analysis.

### Attack Success Rate (ASR)

| Attack | None | SpellCheck | MV@7 | SC+MV@7 |
|--------|------|------------|------|---------|
| BERTattack | 82.5% | 82.2% | **30.3%** | 33.7% |
| PWWS | 87.3% | 87.3% | **36.3%** | 48.1% |
| DeepWordBug | 46.9% | **14.7%** | 26.2% | 17.1% |
| Genetic | 89.9% | 87.7% | 42.8% | **39.7%** |
| VIPER | **26.4%** | 73.8% ❌ | 30.0% | 40.4% |

### Key Findings

1. **MV@7 dominates word-level attacks** (-47 to -52pp) via the noisy oracle mechanism
2. **SpellCheck dominates DeepWordBug** (-32pp); SC+MV is close at -30pp
3. **SC+MV@7 provides broad but non-dominant coverage** — never worst, rarely best
4. **VIPER: SpellCheck is counterproductive** (+47.4pp). SymSpell misinterprets homoglyphs as misspellings and "corrects" them into adversarially useful words. No defense successfully reduces VIPER ASR below baseline.

---

## Comparison Context

| Defense | Source | vs BERTattack | vs DeepWordBug | Utility Cost |
|---------|--------|---------------|----------------|--------------|
| SpellCheck | Exp 1 | -0.2pp | **-32.2pp** | +0.4% |
| MV@7 (noisy oracle) | Exp 1/3 | **-52.2pp** | -20.7pp | -0.4% |
| MV@7 (clean prob, discarded) | Exp 2 | -41.4pp | -9.9pp | -0.4% |
| LabelFlip@0.10 | Exp 2 | -44.7pp | -14.9pp | -4.7% |
| SpellCheck+MV@7 | **Exp 3** | -48.8pp | -29.8pp | **0.0%** |

---

## Fixed Attacker Benchmark

Starting from experiment-3, all experiments use the **same 5 attackers** as a fixed evaluation suite:
- `BERTattack`, `PWWS`, `DeepWordBug`, `Genetic`, `VIPER`

This ensures comparability across experiments and prevents cherry-picking attackers.
