# Branch: experiment-2.1/discretize-probabilities

## Overview

**Purpose**: Implement a discretized probability output defense (Experiment 2.1) that removes all gradient information from the attacker's oracle by returning hard one-hot labels instead of soft probabilities.

**Parent Branch**: `main` (merged from `experiment-5/xarello`)

**Classification**: Output defense (extension of Experiment 2)

**Status**: Complete

---

## Summary of Changes vs Main

This branch adds **one output defense** that takes the extreme end of the output-perturbation spectrum — instead of adding noise to probabilities, it completely eliminates the continuous signal:

### Key Addition

**Discretized Probability Defense (Exp 2.1)**
- `get_prob()` returns hard one-hot `[1,0]` or `[0,1]` based on the victim's argmax
- All soft information (e.g. `[0.9, 0.1]` vs `[0.6, 0.4]`) is destroyed
- Prediction (`get_pred()` = argmax of `get_prob()`) is identical to the undefended victim
- Clean accuracy is exactly unchanged — no utility cost

---

## Architecture

```
Input Defenses (Exp 1)              Output Defenses (Exp 2 + 2.1)
──────────────────────              ──────────────────────────────

[Input] ──► Transform ──► Victim ──► [Output]
                                       │
                                       ▼
                             [Input] ──► Victim ──► Discretize ──►

                             Victim output: [0.87, 0.13]
                             Defense output: [1.0, 0.0]   (hard one-hot)
```

**Why this matters for attackers**

Word-level attackers (BERTattack, PWWS, Genetic) search for word substitutions by comparing probability changes:
- With soft probs: `replace("good") → [0.6, 0.4]` vs `replace("bad") → [0.3, 0.7]` → clear gradient
- With discretized: `replace("good") → [1, 0]` vs `replace("bad") → [1, 0]` → no signal until a flip occurs

The attacker can only observe binary label flips. Until a substitution actually crosses the decision boundary, it appears completely ineffective. This makes gradient-based search near-blind.

**No VIPER**: VIPER is a character-level visual perturbation attack. It does not use `get_prob()` for gradient estimation — it operates by inserting visually similar Unicode characters and checking if the label flips. Discretization provides no additional protection vs. VIPER beyond the undefended baseline.

---

## Files Changed

| File | Change |
|------|--------|
| `defenses/preprocessing.py` | Added `DiscretizedProbabilityDefense(OutputDefenseWrapper)` class |
| `defenses/preprocessing.py` | Registered `'discretize'` / `'disc'` in `get_defense()` factory |
| `runs/eval_defense_accuracy.py` | Added `('discretize', 0.0)` to `DEFENSE_CONFIGS` |
| `scripts/run_experiment-2.1_discretize.sh` | New experiment automation script |
| `explanations/experiment-2.1_discretize-probabilities.md` | This file |

---

## Usage

```bash
# Run full experiment
./scripts/run_experiment-2.1_discretize.sh

# Single attacker with discretize defense
python runs/attack.py PR2 false BERTattack BiLSTM data/PR2 data/PR2/BiLSTM-512.pth \
    results/experiment-2.1_discretize \
    --defense discretize

# Clean accuracy (should be identical to no-defense)
python runs/eval_defense_accuracy.py PR2 BiLSTM data/PR2 data/PR2/BiLSTM-512.pth \
    results/experiment-2.1_discretize
```

---

## Comparison with Experiment 2 Defenses

| Defense | Clean Accuracy | Oracle Signal | Mechanism |
|---------|---------------|---------------|-----------|
| No defense | 100% | Full soft probs | Baseline |
| Label Flip (p=0.1) | ~90% | Noisy labels | Random label flip |
| Confidence Noise (σ=0.1) | ~100% | Noisy soft probs | Gaussian noise |
| **Discretize** | **100%** | **Binary only** | **Hard one-hot** |

Discretization is the most information-limiting of all output defenses: it provides NO soft signal while maintaining perfect clean accuracy.

---

## Expected Hypotheses

1. **Discretize significantly reduces ASR for word-level attacks** (BERTattack, PWWS, Genetic) — these rely heavily on probability gradients
2. **DeepWordBug less affected** — it's char-level and mostly uses get_pred() (binary prediction) already
3. **No clean accuracy drop** — the prediction is identical to undefended victim (argmax unchanged)
4. **Comparison with SC+MV**: Does removing gradient info alone (discretize) match or exceed the protection from SC+MV@3?

---

## Results

See `results/experiment-2.1_discretize/SUMMARY_discretize.md`.
