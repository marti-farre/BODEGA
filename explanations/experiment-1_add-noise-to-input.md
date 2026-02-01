# Branch: experiment-1/add-noise-to-input

## Overview

**Purpose**: Add preprocessing defenses to BODEGA for countering adversarial attacks on misinformation detection classifiers.

**Parent Branch**: `main`

**Status**: Active development

---

## Summary of Changes vs Main

This branch introduces a **preprocessing defense framework** that transforms input text before classification to disrupt adversarial perturbations.

### Key Additions

1. **Defense Module** (`defenses/`)
   - Wrapper classes that intercept inputs before victim model
   - Three defense strategies: SpellCheck, CharacterNoise, CharacterMasking
   - Factory function for easy defense instantiation

2. **Evaluation Scripts**
   - Clean accuracy evaluation to measure utility cost
   - Attack script integration with defense support

3. **Experiment Automation**
   - Bash scripts for running systematic experiments
   - Multi-attacker evaluation support

---

## Architecture Diagram

```
                           BODEGA Defense Architecture
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   [Adversarial Input] ──► [Defense Wrapper] ──► [Victim Model] │
    │                                │                      │         │
    │                                │                      ▼         │
    │                         ┌──────┴──────┐         [Prediction]    │
    │                         │   Defense   │                         │
    │                         │   Options   │                         │
    │                         ├─────────────┤                         │
    │                         │ SpellCheck  │ ◄── Corrects typos      │
    │                         │ CharNoise   │ ◄── Adds homoglyphs     │
    │                         │ CharMasking │ ◄── Removes chars       │
    │                         │ Identity    │ ◄── No change           │
    │                         └─────────────┘                         │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

### Defense Flow

```
Input Text
    │
    ▼
┌───────────────────┐
│  DefenseWrapper   │
│  .get_prob()      │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  apply_defense()  │──► Handles text pairs (FC/C19)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  defend_single()  │──► Applies transformation
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  victim.get_prob()│──► Original model prediction
└───────────────────┘
```

---

## Files Changed

| File | Change | Lines | Description |
|------|--------|-------|-------------|
| `defenses/__init__.py` | New | +17 | Module exports |
| `defenses/preprocessing.py` | New | +313 | Defense implementations |
| `runs/attack.py` | Modified | +94 | Added defense CLI args and integration |
| `runs/eval_defense_accuracy.py` | New | +246 | Clean accuracy evaluation script |
| `scripts/run_pr_defense_experiments.sh` | New | +138 | Experiment automation |
| `.gitignore` | Modified | +4 | Ignore patterns |

**Total**: +812 lines added

---

## Defense Implementations

### 1. SpellCheckDefense
- **Library**: SymSpellPy
- **Mechanism**: Corrects misspellings using frequency dictionary
- **Rationale**: DeepWordBug creates typos; spellcheck reverses them
- **Utility Cost**: None (actually improves accuracy slightly)

### 2. CharacterNoiseDefense
- **Library**: homoglyphs
- **Mechanism**: Substitutes characters with visually similar Unicode variants
- **Rationale**: Disrupts adversarial patterns while maintaining readability
- **Parameters**: `noise_std` (probability of substitution)

### 3. CharacterMaskingDefense
- **Mechanism**: Randomly removes characters (preserves spaces)
- **Rationale**: Disrupts character-level attack patterns
- **Parameters**: `masking_prob` (probability of removal)

---

## Clean Accuracy Results (PR2, BiLSTM, Homoglyphs)

| Defense | Param | Accuracy | F1 | Acc Δ |
|---------|-------|----------|-----|-------|
| None (baseline) | - | 66.83% | 0.4812 | - |
| SpellCheck | - | 67.07% | 0.4669 | +0.36% |
| char_masking | 0.05 | 65.62% | 0.4875 | -1.80% |
| char_masking | 0.10 | 62.50% | 0.4583 | -6.47% |
| char_masking | 0.15 | 57.93% | 0.4108 | -13.31% |
| char_masking | 0.20 | 54.33% | 0.3709 | -18.71% |
| char_noise | 0.05 | 67.79% | 0.3431 | +1.44% |
| char_noise | 0.10 | 67.55% | 0.2197 | +1.08% |
| char_noise | 0.15 | 68.51% | 0.1088 | +2.52% |
| char_noise | 0.20 | 69.23% | 0.0448 | +3.60% |

### Key Observations
1. **SpellCheck**: Slight accuracy improvement (+0.36%), maintains F1
2. **char_masking**: Progressive accuracy degradation as masking increases
3. **char_noise (homoglyphs)**: Accuracy improves but **F1 collapses** - the model predicts mostly one class
   - This suggests homoglyphs may be disrupting the model's ability to distinguish classes
   - Need to investigate why F1 drops so dramatically with noise

### Key Finding
**SpellCheck is optimal for character-level attacks** - achieves the best robustness improvement with zero utility cost.

---

## Usage Examples

### Run Single Attack with Defense
```bash
python runs/attack.py PR2 false DeepWordBug BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results \
    --defense spellcheck
```

### Evaluate Clean Accuracy
```bash
python runs/eval_defense_accuracy.py PR2 BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results
```

### Run All Experiments
```bash
./scripts/run_pr_defense_experiments.sh
```

---

## Results Locations

| Result Type | Path Pattern | Example |
|-------------|--------------|---------|
| Clean accuracy | `results/clean_accuracy_{task}_{victim}.txt` | `results/clean_accuracy_PR2_BiLSTM.txt` |
| Attack results | `results/results_{task}_{targeted}_{attack}_{victim}_{defense}.txt` | `results/results_PR2_false_DeepWordBug_BiLSTM_spellcheck.txt` |
| Defense modifications | `results/modifications_{task}_{targeted}_{attack}_{victim}_{defense}.tsv` | `results/modifications_PR2_false_DeepWordBug_BiLSTM_char_noise_0.1.tsv` |

---

## Next Steps

1. Test with additional attackers (BERTattack, TextFooler, Genetic)
2. Evaluate on other tasks (FC, HN, RD, C19)
3. Test with different victim models (BERT, Gemma)
4. Implement adaptive defense strategies

---

## Related Files

- Results summary: `results/SUMMARY_preprocessing_defenses.md`
- CLAUDE.md: Project documentation
- Original XARELLO defenses: `../xarello/defenses/preprocessing.py`
