# Branch: experiment-2/add-noise-to-output

## Overview

**Purpose**: Implement output perturbation defenses that disturb how attacker models learn from the defender's responses. Unlike input defenses (Experiment 1), these defenses perturb the **output probabilities/labels** returned by the victim model.

**Parent Branch**: `main` (after merge from `experiment-1/implement-survey-defenses`)

**Status**: Implementation complete, pending experiments

---

## Summary of Changes vs Main

This branch adds **three output perturbation defenses** that target the attacker's learning signal rather than the input text:

### Key Additions

1. **Label Flipping Defense (Exp 2a)**
   - Flips output label with probability P(ε)
   - Introduces noise into attacker's reward signal

2. **Random Threshold Defense (Exp 2b)**
   - Uses random decision threshold instead of fixed 0.5
   - Makes decision boundary fuzzy and harder to target

3. **Confidence Perturbation Defense (Exp 2c)**
   - Adds Gaussian noise to output probabilities
   - Obscures true confidence levels

4. **Experiment Script**
   - `run_experiment-2_output-defenses.sh` for systematic evaluation
   - Tests against BERTattack, PWWS, DeepWordBug, Genetic, VIPER

---

## Architecture Diagram

```
                    Output Perturbation Defense Architecture
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  INPUT DEFENSES (Exp 1)              OUTPUT DEFENSES (Exp 2)        │
│  ──────────────────────              ──────────────────────         │
│                                                                     │
│  [Input] ──► Transform ──► Victim ──► [Output]                      │
│              ▲                         │                            │
│              │                         ▼                            │
│         SpellCheck              [Input] ──► Victim ──► Perturb ──►  │
│         CharNoise                                      ▲            │
│         Unicode                                        │            │
│         MajorityVote                             LabelFlip          │
│                                                  RandThreshold       │
│                                                  ConfidenceNoise    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  LABEL FLIPPING (Exp 2a)                                            │
│  ───────────────────────                                            │
│                                                                     │
│  [Input] ──► Victim ──► [P(0), P(1)] ──► Flip with P(ε) ──► Output  │
│                                          │                          │
│                                          ▼                          │
│                              If flip: [P(1), P(0)] (swap probs)     │
│                              Else:    [P(0), P(1)] (keep original)  │
│                                                                     │
│  Effect: Attacker sees inconsistent labels for same/similar inputs  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  RANDOM THRESHOLD (Exp 2b)                                          │
│  ─────────────────────────                                          │
│                                                                     │
│  [Input] ──► Victim ──► P(class=1) ──► Compare to random threshold  │
│                                        │                            │
│                                        ▼                            │
│                         threshold = 0.5 + uniform(-ε, +ε)           │
│                         pred = 1 if P(1) >= threshold else 0        │
│                                                                     │
│  Effect: Samples near boundary get inconsistent predictions         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  CONFIDENCE PERTURBATION (Exp 2c)                                   │
│  ────────────────────────────────                                   │
│                                                                     │
│  [Input] ──► Victim ──► [P(0), P(1)] ──► Add Gaussian noise         │
│                                          │                          │
│                                          ▼                          │
│                              noise ~ N(0, σ²)                       │
│                              perturbed = clip(probs + noise, 0, 1)  │
│                              output = normalize(perturbed)          │
│                                                                     │
│  Effect: Obscures true confidence, hinders gradient-based attacks   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Defense Integration

```
                     Defense Wrapper Hierarchy

                     ┌─────────────────┐
                     │ OpenAttack      │
                     │ Classifier      │ (Interface)
                     │ ─────────────── │
                     │ get_prob()      │
                     │ get_pred()      │
                     └────────┬────────┘
                              │
           ┌──────────────────┴──────────────────┐
           │                                      │
           ▼                                      ▼
┌─────────────────────┐               ┌─────────────────────┐
│ DefenseWrapper      │ (Exp 1)       │ OutputDefenseWrapper│ (Exp 2)
│ ─────────────────── │               │ ─────────────────── │
│ Transforms INPUT    │               │ Transforms OUTPUT   │
│ before victim call  │               │ after victim call   │
│                     │               │                     │
│ defend_single()     │               │ perturb_prob()      │
│ apply_defense()     │               │                     │
└────────┬────────────┘               └────────┬────────────┘
         │                                      │
    ┌────┴────┐                        ┌───────┼───────┐
    │         │                        │       │       │
    ▼         ▼                        ▼       ▼       ▼
SpellCheck  Unicode              LabelFlip  Random  Confidence
CharNoise   MajorityVote                    Thresh   Noise
CharMask    Identity
```

---

## Files Changed

| File | Change | Lines | Description |
|------|--------|-------|-------------|
| `defenses/preprocessing.py` | Modified | +150 | Added OutputDefenseWrapper, LabelFlippingDefense, RandomThresholdDefense, ConfidencePerturbationDefense |
| `runs/eval_defense_accuracy.py` | Modified | +9 | Added output defenses to DEFENSE_CONFIGS |
| `scripts/run_experiment-2_output-defenses.sh` | New | +75 | Experiment automation script |
| `explanations/experiment-2_add-noise-to-output.md` | New | ~250 | This documentation |

**Total**: ~250 lines added/modified

---

## Defense Implementations

### 1. LabelFlippingDefense (Exp 2a)

**Concept**: Introduce noise into the attacker's reward signal by randomly flipping predictions.

**Mechanism**:
1. Get prediction from victim model
2. With probability `flip_prob`, swap the class probabilities: `[P(0), P(1)] → [P(1), P(0)]`
3. Return (potentially flipped) prediction

**Rationale**:
- Attackers learn which perturbations are "successful" by observing label changes
- If labels are randomly flipped, the attacker gets inconsistent feedback
- This makes it harder to learn an effective attack policy

**Parameters**:
- `flip_prob`: Probability of flipping (0.05, 0.10, 0.15)

**Utility Cost**: Expected `flip_prob` × 100% accuracy drop (e.g., 10% flip = ~10% accuracy drop on clean data)

**Best Against**: Reinforcement learning-based attackers that learn from feedback

### 2. RandomThresholdDefense (Exp 2b)

**Concept**: Make the decision boundary fuzzy by using a random threshold per query.

**Mechanism**:
1. Get probability P(class=1) from victim
2. Sample random threshold: `threshold = 0.5 + uniform(-ε, +ε)`
3. Predict class=1 if P(1) >= threshold, else class=0

**Rationale**:
- Attackers often try to find inputs that just barely cross the decision boundary
- With a random threshold, the same input may get different predictions
- This makes it impossible to precisely target the boundary

**Parameters**:
- `threshold_range` (ε): Range for threshold variation (0.05, 0.10, 0.15)

**Utility Cost**: Expected ~0% for confident predictions, higher for borderline cases

**Best Against**: Attacks that probe near the decision boundary

### 3. ConfidencePerturbationDefense (Exp 2c)

**Concept**: Obscure true confidence levels by adding noise to probabilities.

**Mechanism**:
1. Get probabilities from victim: `[P(0), P(1)]`
2. Add Gaussian noise: `perturbed = probs + N(0, σ²)`
3. Clip to [0.01, 0.99] and renormalize to sum to 1

**Rationale**:
- Gradient-based attacks (BERTattack) rely on precise probability estimates
- Noisy probabilities provide less useful gradient information
- This can reduce attack effectiveness without changing many predictions

**Parameters**:
- `noise_std` (σ): Standard deviation of Gaussian noise (0.05, 0.10, 0.15)

**Utility Cost**: Expected ~0-5% for moderate noise levels

**Best Against**: Gradient-based attacks (BERTattack, BAE)

---

## Usage Examples

### Factory Function
```python
from defenses.preprocessing import get_defense

# Label flipping (10% flip probability)
flip_def = get_defense('label_flip', victim, param=0.10, seed=42)

# Random threshold (±0.1 range)
thresh_def = get_defense('random_threshold', victim, param=0.10, seed=42)

# Confidence noise (σ=0.10)
noise_def = get_defense('confidence_noise', victim, param=0.10, seed=42)

# Alternative aliases
flip_def = get_defense('flip', victim, param=0.10)
thresh_def = get_defense('threshold', victim, param=0.10)
noise_def = get_defense('conf_noise', victim, param=0.10)
```

### Command Line
```bash
# Attack with label flipping defense
python runs/attack.py PR2 false BERTattack BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results \
    --defense label_flip --defense_param 0.10 --defense_seed 42

# Attack with random threshold defense
python runs/attack.py PR2 false DeepWordBug BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results \
    --defense random_threshold --defense_param 0.15 --defense_seed 42

# Attack with confidence noise defense
python runs/attack.py PR2 false BERTattack BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results \
    --defense confidence_noise --defense_param 0.10 --defense_seed 42
```

### Run All Output Defense Experiments
```bash
./scripts/run_experiment-2_output-defenses.sh
```

---

## Expected Results

### Clean Accuracy Impact

| Defense | Param | Expected Acc Δ | Notes |
|---------|-------|----------------|-------|
| label_flip | 0.05 | -5% | Proportional to flip_prob |
| label_flip | 0.10 | -10% | Higher noise, more confusion |
| label_flip | 0.15 | -15% | Maximum tested |
| random_threshold | 0.05 | ~0% | Only affects borderline cases |
| random_threshold | 0.10 | -1 to -3% | Moderate boundary fuzziness |
| random_threshold | 0.15 | -2 to -5% | Wide threshold range |
| confidence_noise | 0.05 | ~0% | Predictions rarely change |
| confidence_noise | 0.10 | -1 to -3% | Moderate noise |
| confidence_noise | 0.15 | -3 to -5% | Higher noise |

### Attack Success Rate Reduction

| Defense | vs BERTattack | vs PWWS | vs DeepWordBug | Notes |
|---------|---------------|---------|----------------|-------|
| label_flip@0.10 | 10-20pp | 10-20pp | 10-20pp | Uniform reduction |
| random_threshold@0.10 | 5-10pp | 5-10pp | 5-10pp | Best for boundary probing |
| confidence_noise@0.10 | **15-25pp** | 5-10pp | 0-5pp | Best for gradient attacks |

**Key Expectations**:
- **Label flipping**: Uniform reduction but high utility cost
- **Random threshold**: Low utility cost, moderate ASR reduction
- **Confidence noise**: Strong against gradient-based attacks (BERTattack)

---

## Results Directory Structure

```
results/
├── experiment-1_add-noise-to-input/       # Previous experiment results
├── experiment-1_implement-survey-defenses/ # Survey defenses results
│
└── experiment-2_add-noise-to-output/       # NEW: Output defense results
    ├── SUMMARY_output_defenses.md
    ├── clean_accuracy_PR2_BiLSTM.txt
    ├── results_PR2_False_BERTattack_BiLSTM_none.txt
    ├── results_PR2_False_BERTattack_BiLSTM_label_flip_0.05.txt
    ├── results_PR2_False_BERTattack_BiLSTM_label_flip_0.1.txt
    ├── results_PR2_False_BERTattack_BiLSTM_label_flip_0.15.txt
    ├── results_PR2_False_BERTattack_BiLSTM_random_threshold_0.05.txt
    ├── results_PR2_False_BERTattack_BiLSTM_random_threshold_0.1.txt
    ├── results_PR2_False_BERTattack_BiLSTM_random_threshold_0.15.txt
    ├── results_PR2_False_BERTattack_BiLSTM_confidence_noise_0.05.txt
    ├── results_PR2_False_BERTattack_BiLSTM_confidence_noise_0.1.txt
    ├── results_PR2_False_BERTattack_BiLSTM_confidence_noise_0.15.txt
    └── ... (same pattern for PWWS, DeepWordBug, Genetic, VIPER)
```

---

## Comparison: Input vs Output Defenses

| Aspect | Input Defenses (Exp 1) | Output Defenses (Exp 2) |
|--------|------------------------|-------------------------|
| **What's modified** | Input text | Output probabilities/labels |
| **When applied** | Before classification | After classification |
| **Computational cost** | 1× (except MajorityVote) | 1× |
| **Visibility to attacker** | Modified text visible | Only predictions visible |
| **Utility impact** | Text degradation | Prediction noise |
| **Best against** | Character-level attacks | Learning-based attacks |

### When to Use Each

- **Input defenses**: When attacks modify the text structure (DeepWordBug, VIPER)
- **Output defenses**: When attacks learn from model feedback (BERTattack, RL-based)
- **Combine both**: For comprehensive protection (may stack utility costs)

---

## Next Steps

1. Run `./scripts/run_experiment-2_output-defenses.sh`
2. Analyze results and compare with input defenses
3. Update `results/experiment-2_add-noise-to-output/SUMMARY_output_defenses.md`
4. Consider combining input + output defenses
5. Evaluate on additional tasks (FC, C19) if time permits

---

## References

- **Thesis Focus**: Preprocessing defenses against adversarial attacks on misinformation detection
- **Previous Experiments**:
  - `explanations/experiment-1_add-noise-to-input.md`
  - `explanations/experiment-1_implement-survey-defenses.md`
- **BODEGA Paper**: "Verifying the Robustness of Automatic Credibility Assessment"
- **Survey Paper**: "A Survey of Adversarial Defenses and Robustness in NLP" (ACM Computing Surveys 2023)
