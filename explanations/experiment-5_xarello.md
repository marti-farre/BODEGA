# Experiment 5: XARELLO Adaptive Attacker vs SC+MV@7 Defense

**Branch (BODEGA):** `experiment-5/xarello`
**Branch (XARELLO):** `experiment-5/bodega-defenses`
**Parent:** `main` (after merging experiment-4)
**Status:** 5.1 complete; 5.2 pending (needs server)
**Date:** March 2026

---

## Overview

All previous experiments (1–4) tested **non-adaptive** attackers: BERTattack, PWWS, DeepWordBug, Genetic, VIPER from OpenAttack. These attackers have no knowledge of the defense — they query the victim's unmodified probabilities during planning, and only see the defended output when they submit.

Experiment 5 tests **XARELLO**, a Reinforcement Learning attacker that *adapts* to its target through repeated interaction. XARELLO uses Q-learning with a Transformer-based Q estimator and learns which character-level edits (swap, insert, delete) cause label flips.

**Key question**: Does SC+MV@7 remain effective against an adversary that can adapt to the defense over thousands of training episodes?

Two sub-experiments:

| Sub-exp | XARELLO training | Evaluation defense | Question |
|---------|------------------|--------------------|----------|
| **5.1** | Undefended victim | SC+MV@7 | Can SC+MV@7 defend against a pre-trained adaptive attacker? |
| **5.2** | Defended victim (SC+MV@7) | SC+MV@7 | Can XARELLO adapt and break SC+MV@7 when trained against it? |

---

## Why SC+MV@7?

From experiment-3, SC+MV@7 is the best broad-coverage defense:
- BERTattack: -48.8pp ASR (82.5% → 33.7%)
- PWWS: -52.4pp (87.5% → 35.1%)
- DeepWordBug: -19.0pp (46.9% → 27.9%)
- Genetic: -52.7pp (89.9% → 37.2%)
- Utility cost: 0.0% accuracy delta

The stochastic MV oracle confuses gradient-based attackers. XARELLO's RL mechanism uses `get_pred()` (NOT `get_prob()`) to check attack success — it observes discrete win/lose signals rather than probability gradients. This makes XARELLO a different threat model.

---

## Architecture

### Experiment 5.1: Pre-trained XARELLO vs SC+MV@7

```
PHASE 1 — Training (undefended):
   victim (undefended)
       ↑
   XARELLO Q-agent (20 epochs × 1600 texts)
       learns char-level edit strategies

PHASE 2 — Evaluation (defended):
   victim
     ↓ SC+MV@7 wrapper
   defended victim
       ↑
   pre-trained XARELLO (frozen Q-model)
       generates adversarial candidates
```

The pre-trained XARELLO doesn't know about the defense. It applies its learned edit policy, but the defended victim may reject perturbations that the original victim accepted.

### Experiment 5.2: Adaptive XARELLO (trained with defense)

```
PHASE 1 — Training (defended):
   victim
     ↓ SC+MV@7 wrapper
   defended victim
       ↑
   XARELLO Q-agent (20 epochs × 1600 texts)
       learns to attack despite SC+MV stochasticity

PHASE 2 — Evaluation (defended):
   same setup as training
```

The adaptive XARELLO learns from the stochastic MV oracle throughout training. If SC+MV@7 ASR in 5.2 > 5.1, the defense can be broken by adaptation. If similar, the defense is robust even to adaptive attackers.

---

## Summary of Changes

### XARELLO: `experiment-5/bodega-defenses`

| File | Change |
|------|--------|
| `defenses/preprocessing.py` | Added `MajorityVoteDefense`, `SpellCheckMVDefense`; updated `get_defense()` |
| `main-train-eval.py` | Added `defense`, `defense_param`, `defense_seed` positional args (positions 7–9) |

### BODEGA: `experiment-5/xarello`

| File | Change |
|------|--------|
| `scripts/run_experiment-5_xarello.sh` | New experiment script (trains + evaluates 5.1 and 5.2) |
| `explanations/experiment-5_xarello.md` | This file |

---

## Usage

```bash
# Run both 5.1 and 5.2 end-to-end (long — trains XARELLO twice)
./scripts/run_experiment-5_xarello.sh

# Manually: train XARELLO without defense (5.1 base model)
cd ../xarello
python main-train-eval.py PR2 BiLSTM ~/data/xarello/models/exp5.1/PR2-BiLSTM

# Manually: evaluate pre-trained XARELLO with SC+MV@7
python evaluation/attack.py \
    --task PR2 --victim BiLSTM \
    --qmodel_path ~/data/xarello/models/exp5.1/PR2-BiLSTM/xarello-qmodel.pth \
    --out_dir results/exp5.1 \
    --defense spellcheck_mv --defense_param 7 --defense_seed 42

# Manually: train XARELLO on defended victim (5.2)
python main-train-eval.py PR2 BiLSTM ~/data/xarello/models/exp5.2/PR2-BiLSTM \
    none 0.0 42 spellcheck_mv 7.0 42

# Manually: evaluate adaptive XARELLO with SC+MV@7
python evaluation/attack.py \
    --task PR2 --victim BiLSTM \
    --qmodel_path ~/data/xarello/models/exp5.2/PR2-BiLSTM/xarello-qmodel.pth \
    --out_dir results/exp5.2 \
    --defense spellcheck_mv --defense_param 7 --defense_seed 42
```

---

## Expected Results

Based on the threat model analysis:

**5.1 (pre-trained XARELLO vs SC+MV@7)**:
- XARELLO's char-level edits (swaps, inserts, deletes) are exactly what SpellCheck is designed to catch
- MV@7 stochastic oracle further disrupts the pre-trained Q-model's learned edit preferences
- Expected: SC+MV@7 provides meaningful ASR reduction (similar to non-adaptive attackers)

**5.2 (adaptive XARELLO vs SC+MV@7)**:
- XARELLO adapts to stochastic oracle: RL agents can learn to exploit consistent long-run win/loss signals even if individual signals are noisy
- SpellCheck limits: if XARELLO learns semantically meaningful substitutions (not just typos), SC won't help
- MV stochasticity: may slow XARELLO's learning (noisier reward signal) but may not prevent convergence
- Expected: higher ASR than 5.1, but unclear if it fully breaks the defense

---

## Results

### 5.1: Pre-trained XARELLO vs SC+MV@7

| Metric | No Defense | SC+MV@7 | Delta |
|--------|-----------|---------|-------|
| ASR | 97.3% (71/73) | 37.7% (26/69) | **−59.6pp** |
| BODEGA score | 0.711 | 0.292 | −0.419 |
| Queries/example | 28.3 | 17.9 | −10.4 |

SC+MV@7 reduces XARELLO's ASR by 59.6pp — the largest drop of any defense–attacker pair tested across all experiments (beating Genetic's 52.7pp drop). SpellCheck corrects nearly all of XARELLO's character-level edits before the MV oracle votes.

See `results/experiment-5_xarello/SUMMARY_xarello.md` for full analysis.

### 5.2: Adaptive XARELLO

Pending — requires server. Training with SC+MV@7 in loop estimated at ~3 weeks on MacBook (7× slower due to MV copies).

---

## Notes on XARELLO Threat Model

XARELLO calls `victim.get_pred()` to check attack success (not `get_prob()`). This means:
- It observes **binary win/lose signals**, not probability gradients
- The stochastic MV oracle affects it differently than gradient-based word-level attackers
- RL can learn from noisy rewards via averaging over episodes — the noise slows but doesn't prevent learning
- The SpellCheck component is XARELLO's main vulnerability: if edits produce recognized typos, they get corrected before MV

Contrast with experiment 4 finding: stochastic oracle (via MV) fully broke BERTattack (gradient-based). XARELLO's RL mechanism is more robust to oracle noise.
