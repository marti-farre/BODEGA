# Experiment 4: Unicode Canonicalization + MajorityVote Defense

**Branch:** `experiment-4/unicode-canonicalization`
**Parent:** `main` (after merging experiment-3)
**Status:** Completed
**Date:** March 2026

---

## Overview

This experiment targets **VIPER**, the one attack that no previous defense could reduce below baseline ASR. VIPER replaces characters with visually identical Unicode confusables (e.g. Cyrillic 'а' → Latin 'a'). Experiment 3 found that SpellCheck *worsened* VIPER ASR from 26.4% to 73.8% (+47.4pp) because SymSpell misidentifies homoglyphs as misspellings and corrects them to adversarially useful words.

### Motivation

The right tool for homoglyph attacks is **Unicode canonicalization**: deterministically map known confusables back to their ASCII equivalents before classification, rather than trusting a spell corrector to guess them.

From experiment-3:
- **MV@7** is the best broad defense for word-level attacks (-47 to -52pp) but barely touches VIPER (+3.6pp — essentially no effect)
- **SpellCheck** directly worsens VIPER (+47.4pp) — counterproductive
- **SC+MV@7** still worse than baseline for VIPER (+14pp)
- No defense reduces VIPER ASR below 26.4% baseline

Unicode canonicalization should directly neutralize the VIPER mechanism. Combining it with MV@7 should maintain word-level robustness.

---

## Summary of Changes vs Main

| File | Change |
|------|--------|
| `defenses/preprocessing.py` | Added `UnicodeMVDefense` class + `'unicode_mv'` in `get_defense()` |
| `runs/eval_defense_accuracy.py` | Added `unicode_mv` configs (3, 5, 7 copies) |
| `scripts/run_experiment-4_unicode-canonicalization.sh` | New experiment script |
| `explanations/experiment-4_unicode-canonicalization.md` | This file |

Note: `UnicodeCanonicalizationDefense` was already implemented in `defenses/preprocessing.py` from the experiment-1 survey. This experiment evaluates it properly and combines it with MV.

---

## Architecture

### UnicodeMVDefense Pipeline

```
Adversarial Input (e.g. VIPER homoglyphs)
       │
       ▼
┌──────────────────────────────┐
│   Unicode Canonicalization   │  ← Map Cyrillic/Greek confusables → ASCII
│                              │    Remove zero-width invisible chars
│   NFKC normalization         │    e.g. 'аdvеrtisе' → 'advertise'
└──────┬───────────────────────┘
       │ canonicalized text
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

### UnicodeCanonicalizationDefense steps

1. **Remove zero-width chars**: ZWSP, ZWNJ, ZWJ, soft hyphens, etc. (20 invisible chars)
2. **NFKC normalization**: handles ligatures, fullwidth chars, compatibility decomposition
3. **Confusables map**: explicit lookup table mapping Cyrillic/Greek/typographic chars to ASCII (e.g. `\u0430 → 'a'`, `\u0435 → 'e'`, `\u043e → 'o'`)

### API Behavior

```
get_prob(adversarial_input):
    canonicalized = UnicodeCanonicalization(adversarial_input)
    # Run N noisy copies → return vote-fraction probabilities (noisy oracle)
    for N copies:
        noisy = add_mv_noise(canonicalized)
        probs.append(victim(noisy))
    return vote_fractions(probs)

get_pred(adversarial_input):
    return get_prob(adversarial_input).argmax()   # majority class
```

### Comparison with Individual Defenses

```
Unicode alone:
  adversarial → Unicode canonicalize → victim → prediction

MajorityVote alone:
  adversarial → [N noisy copies] → victim × N → vote fractions → argmax

UnicodeMV (combined):
  adversarial → Unicode canonicalize → [N noisy copies] → victim × N → vote fractions → argmax
```

---

## Experiment Design

### Defenses Compared

| Defense | Description | Expected Strength |
|---------|-------------|-------------------|
| none | Baseline | - |
| unicode | Unicode canonicalization alone | VIPER, homoglyph attacks |
| majority_vote@7 | MV alone (noisy oracle) | Word-level attacks |
| unicode_mv@7 | Combined | Both word-level + homoglyph |

### Fixed Attacker Benchmark

All experiments use the same 5 attackers for comparability:

| Attacker | Type | Notes |
|----------|------|-------|
| BERTattack | Word-level | Semantic substitutions |
| PWWS | Word-level | Saliency-weighted |
| DeepWordBug | Char-level | Typos (swaps, inserts, deletes) |
| Genetic | Word-level | Evolutionary |
| VIPER | Char-level | Visual homoglyphs ← primary target |

---

## Usage

```bash
# Run all experiment-4 experiments
./scripts/run_experiment-4_unicode-canonicalization.sh

# Run individual attack with unicode_mv defense
python3 runs/attack.py PR2 false VIPER BiLSTM \
    data/PR2 data/PR2/BiLSTM-512.pth results/experiment-4_unicode-canonicalization \
    --defense unicode_mv --defense_param 7 --defense_seed 42

# Clean accuracy evaluation (includes unicode_mv)
python3 runs/eval_defense_accuracy.py PR2 BiLSTM data/PR2 data/PR2/BiLSTM-512.pth \
    results/experiment-4_unicode-canonicalization
```

---

## Results

See `results/experiment-4_unicode-canonicalization/SUMMARY_unicode_defenses.md` for the full analysis.

### Attack Success Rate (ASR)

| Attack | None | Unicode | MV@7 | UC+MV@7 |
|--------|------|---------|------|---------|
| BERTattack | 82.5% | — | — | — |
| PWWS | 87.3% | — | — | — |
| DeepWordBug | 46.9% | — | — | — |
| Genetic | 89.9% | — | — | — |
| VIPER | 26.4% | — | — | — |

*(To be filled after experiments complete)*

---

## Comparison Context

| Defense | Source | vs BERTattack | vs VIPER | Utility Cost |
|---------|--------|---------------|----------|--------------|
| SpellCheck | Exp 1 | -0.2pp | **+47.4pp** ❌ | +0.4% |
| MV@7 | Exp 3 | **-52.2pp** | +3.6pp ❌ | -0.4% |
| SpellCheck+MV@7 | Exp 3 | -48.8pp | +14.0pp ❌ | 0.0% |
| Unicode | **Exp 4** | — | — | — |
| Unicode+MV@7 | **Exp 4** | — | — | — |

*(To be filled after experiments complete)*

---

## Fixed Attacker Benchmark

Starting from experiment-3, all experiments use the **same 5 attackers** as a fixed evaluation suite:
- `BERTattack`, `PWWS`, `DeepWordBug`, `Genetic`, `VIPER`

This ensures comparability across experiments and prevents cherry-picking attackers.
