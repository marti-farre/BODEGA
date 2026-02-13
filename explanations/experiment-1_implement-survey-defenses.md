# Branch: experiment-1/implement-survey-defenses

## Overview

**Purpose**: Implement two additional preprocessing defenses based on the survey paper "A Survey of Adversarial Defenses and Robustness in NLP" (ACM Computing Surveys 2023).

**Parent Branch**: `main` (after merge from `experiment-1/add-noise-to-input`)

**Status**: Implementation complete, pending experiments

---

## Summary of Changes vs Main

This branch adds **two survey-based preprocessing defenses** that complement the existing defenses without modifying victim models:

### Key Additions

1. **Unicode Canonicalization Defense**
   - Removes zero-width/invisible characters
   - Applies NFKC Unicode normalization
   - Maps confusable characters (Cyrillic, Greek) to ASCII equivalents

2. **Majority Vote Defense**
   - Creates N perturbed copies of input
   - Classifies each copy independently
   - Returns majority vote as final prediction

3. **Experiment Script**
   - `run_survey_defense_experiments.sh` for systematic evaluation
   - Tests against DeepWordBug, VIPER, TextFooler

---

## Architecture Diagram

```
                    Survey Defense Architecture
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  UNICODE CANONICALIZATION                                       │
│  ────────────────────────                                       │
│                                                                 │
│  [Input] ──► Remove Zero-Width ──► NFKC Normalize ──► Map      │
│              Characters            (decompose +     Confusables │
│              (ZWSP, ZWJ, etc.)     compose)        to ASCII     │
│                                                                 │
│  Example: "Hеllo\u200bWorld" ──► "HelloWorld"                   │
│           (Cyrillic е + ZWSP)    (ASCII e, no ZWSP)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  MAJORITY VOTE                                                  │
│  ────────────                                                   │
│                                                                 │
│  [Input] ──┬──► Perturb Copy 1 ──► Victim ──► Pred 1 ─┐        │
│            ├──► Perturb Copy 2 ──► Victim ──► Pred 2 ─┼──► Vote │
│            ├──► Perturb Copy 3 ──► Victim ──► Pred 3 ─┤        │
│            ├──► ...              ...        ...       │        │
│            └──► Perturb Copy N ──► Victim ──► Pred N ─┘        │
│                                                                 │
│  Perturbations: char swap, delete, insert (random)              │
│  Aggregation: hard voting (count) or soft voting (avg probs)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Defense Integration

```
                     DefenseWrapper Hierarchy

                     ┌─────────────────┐
                     │ DefenseWrapper  │ (ABC)
                     │ ─────────────── │
                     │ get_prob()      │──► calls apply_defense() + victim
                     │ apply_defense() │──► handles text pairs
                     │ defend_single() │──► abstract method
                     └────────┬────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ SpellCheck    │   │ Unicode         │   │ MajorityVote    │
│ CharNoise     │   │ Canonicalization│   │ ────────────    │
│ CharMasking   │   │ ────────────────│   │ OVERRIDES       │
│ Identity      │   │ defend_single() │   │ get_prob()      │
│ ─────────     │   │ only            │   │ (needs N model  │
│ defend_single │   │                 │   │  queries)       │
│ only          │   │                 │   │                 │
└───────────────┘   └─────────────────┘   └─────────────────┘
```

---

## Files Changed

| File | Change | Lines | Description |
|------|--------|-------|-------------|
| `defenses/preprocessing.py` | Modified | +200 | Added UnicodeCanonicalizationDefense, MajorityVoteDefense |
| `defenses/__init__.py` | Modified | +2 | Export new classes |
| `runs/eval_defense_accuracy.py` | Modified | +4 | Added unicode and majority_vote to DEFENSE_CONFIGS |
| `scripts/run_survey_defense_experiments.sh` | New | +65 | Experiment automation script |
| `scripts/run_pr_defense_experiments.sh` | Modified | +1 | Updated OUT_DIR path |
| `scripts/run_multi_attacker_experiments.sh` | Modified | +1 | Updated OUT_DIR path |
| `explanations/experiment-1_implement-survey-defenses.md` | New | ~200 | This documentation |

**Total**: ~270 lines added/modified

---

## Defense Implementations

### 1. UnicodeCanonicalizationDefense

**Survey Reference**: Section 5.1.2 - Perturbation Correction (Bhalerao et al.)

**Mechanism**:
1. **Remove zero-width characters** (22 types)
   - Zero Width Space (U+200B)
   - Zero Width Joiner/Non-Joiner (U+200C, U+200D)
   - Byte Order Mark (U+FEFF)
   - Various language-specific fillers

2. **Apply NFKC normalization**
   - Compatibility decomposition + canonical composition
   - Handles fullwidth chars, ligatures, etc.

3. **Map confusables to ASCII** (60+ mappings)
   - Cyrillic: а→a, е→e, о→o, р→p, с→c, etc.
   - Greek: α→a, ε→e, ο→o, etc.
   - Typographic: smart quotes→ASCII quotes, dashes→hyphen

**Rationale**: Character-level attacks like VIPER use homoglyphs and invisible characters. This defense normalizes them back to standard ASCII.

**Parameters**: None required (all transformations are deterministic)

**Utility Cost**: Expected ~0% (normalization should not affect semantic content)

### 2. MajorityVoteDefense

**Survey Reference**: Section 5.1.1 - Perturbation Identification (Swenor & Kalita)

**Mechanism**:
1. Generate N perturbed copies of input
2. Each copy has random character-level perturbations:
   - **Delete**: Remove character (prob 1/3)
   - **Swap**: Swap adjacent characters (prob 1/3)
   - **Insert**: Insert random lowercase letter (prob 1/3)
3. Classify all N copies with victim model
4. Aggregate predictions via voting

**Aggregation Methods**:
- **Hard voting** (default): Count predictions, return proportions
- **Soft voting**: Average probability distributions

**Rationale**: Adversarial perturbations are fragile - small random changes can "break" them. By voting across multiple perturbed versions, we can "vote out" the adversarial effect.

**Parameters**:
- `num_copies`: Number of perturbed copies (default: 5, recommend odd numbers)
- `perturbation_prob`: Probability of perturbing each character (default: 0.1)
- `aggregation`: 'hard' or 'soft' (default: 'hard')

**Utility Cost**: Expected 2-5% accuracy drop due to perturbations

**Computational Cost**: N× model queries (5 copies = 5× inference time)

---

## Usage Examples

### Factory Function
```python
from defenses.preprocessing import get_defense

# Unicode canonicalization
unicode_def = get_defense('unicode', victim, verbose=True)

# Majority vote with 5 copies
vote_def = get_defense('majority_vote', victim, param=5, seed=42)

# Alternative aliases
vote_def = get_defense('vote', victim, param=7)
```

### Command Line
```bash
# Attack with unicode defense
python runs/attack.py PR2 false VIPER BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results \
    --defense unicode --verbose

# Attack with majority vote (5 copies)
python runs/attack.py PR2 false DeepWordBug BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results \
    --defense majority_vote --defense_param 5 --defense_seed 42
```

### Run All Survey Defense Experiments
```bash
./scripts/run_survey_defense_experiments.sh
```

---

## Expected Results

### Clean Accuracy Impact

| Defense | Expected Acc Δ | Notes |
|---------|----------------|-------|
| unicode | ~0% | Deterministic normalization |
| majority_vote@3 | -1 to -3% | Fewer copies, less noise |
| majority_vote@5 | -2 to -5% | Balanced |
| majority_vote@7 | -3 to -7% | More copies, more noise |

### Attack Success Rate Reduction

| Defense | vs DeepWordBug | vs VIPER | vs TextFooler |
|---------|----------------|----------|---------------|
| unicode | 5-15pp | **20-40pp** | 0-5pp |
| majority_vote@5 | 15-25pp | 15-25pp | 10-20pp |

**Key Expectations**:
- Unicode defense should be **highly effective against VIPER** (visual perturbations)
- Majority vote should provide **consistent reduction across all attack types**
- Unicode has **zero utility cost**, majority vote has **moderate utility cost**

---

## Results Directory Structure

```
results/
├── experiment-1_add-noise-to-input/       # Previous experiment results
│   ├── SUMMARY_preprocessing_defenses.md
│   ├── clean_accuracy_PR2_BiLSTM.txt
│   └── results_*.txt, modifications_*.tsv
│
└── experiment-1_implement-survey-defenses/ # New experiment results
    ├── clean_accuracy_PR2_BiLSTM.txt
    ├── results_PR2_False_DeepWordBug_BiLSTM.txt
    ├── results_PR2_False_DeepWordBug_BiLSTM_unicode.txt
    ├── results_PR2_False_DeepWordBug_BiLSTM_majority_vote_3.txt
    ├── results_PR2_False_DeepWordBug_BiLSTM_majority_vote_5.txt
    ├── results_PR2_False_DeepWordBug_BiLSTM_majority_vote_7.txt
    ├── results_PR2_False_VIPER_BiLSTM*.txt
    └── results_PR2_False_TextFooler_BiLSTM*.txt
```

---

## Comparison with Previous Defenses

| Defense | Type | Utility Cost | Best Against | Computational Cost |
|---------|------|--------------|--------------|-------------------|
| SpellCheck | Correction | ~0% | DeepWordBug | 1× |
| char_masking | Perturbation | 2-7% | All (moderate) | 1× |
| char_noise | Perturbation | ~0% (but F1 collapse) | All (high) | 1× |
| **unicode** | Normalization | ~0% | VIPER, homoglyphs | 1× |
| **majority_vote** | Ensemble | 2-5% | All (consistent) | N× |

---

## Next Steps

1. Run `./scripts/run_survey_defense_experiments.sh`
2. Analyze results and compare with previous defenses
3. Update `results/experiment-1_implement-survey-defenses/SUMMARY_survey_defenses.md`
4. Upload results to Google Drive for supervisor review
5. Consider combining defenses (e.g., unicode + majority_vote)

---

## References

- **Survey Paper**: "A Survey of Adversarial Defenses and Robustness in NLP" (ACM Computing Surveys 2023)
  - Section 5.1.1: Perturbation Identification (Majority Vote)
  - Section 5.1.2: Perturbation Correction (Unicode Canonicalization)
- **Unicode Confusables**: https://www.unicode.org/Public/security/latest/confusables.txt
- **Previous Branch**: `explanations/experiment-1_add-noise-to-input.md`
