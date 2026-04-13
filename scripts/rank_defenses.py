#!/usr/bin/env python3
"""Rank defenses by accuracy-penalized BODEGA score.

Reads BLEURT results and clean accuracy results to compute:
  effective_score = clean_accuracy * (1 - avg_BODEGA)

Higher effective_score = better defense (low BODEGA + high accuracy).

Usage:
    cd ~/BODEGA && python scripts/rank_defenses.py [VICTIM]

    VICTIM: BiLSTM (default), BERT, or GEMMA
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

VICTIM = sys.argv[1] if len(sys.argv) > 1 else 'BiLSTM'
RESULTS_DIR = Path('results/experiment-7_bleurt')
ACCURACY_DIR = Path('results/clean_accuracy')

TASKS = ['PR2', 'FC', 'HN', 'RD']
ATTACKERS = ['BERTattack', 'PWWS', 'DeepWordBug', 'Genetic']


def parse_bodega_score(filepath):
    """Extract BODEGA score from result file."""
    try:
        with open(filepath) as f:
            for line in f:
                if line.startswith('BODEGA score:'):
                    return float(line.split(':')[1].strip())
    except (FileNotFoundError, ValueError):
        pass
    return None


def parse_clean_accuracy(filepath):
    """Extract accuracy from clean accuracy result file."""
    try:
        with open(filepath) as f:
            for line in f:
                if line.startswith('Accuracy:'):
                    return float(line.split(':')[1].strip())
    except (FileNotFoundError, ValueError):
        pass
    return None


def get_defense_key(defense, param):
    """Build human-readable defense key."""
    if defense == 'none':
        return 'none'
    if param in (0.0, 0, '0', '0.0'):
        return defense
    return f'{defense}_{param}'


# Collect all BODEGA scores per defense
bodega_scores = defaultdict(list)  # defense_key -> list of scores
bodega_by_task = defaultdict(lambda: defaultdict(list))  # defense_key -> task -> scores

for fname in sorted(RESULTS_DIR.glob(f'results_*_{VICTIM}*.txt')):
    name = fname.stem  # e.g., results_PR2_False_BERTattack_BiLSTM_spellcheck
    parts = name.split('_')

    # Parse: results_TASK_TARGETED_ATTACKER_VICTIM[_DEFENSE[_PARAM]]
    # Find victim position
    try:
        victim_idx = parts.index(VICTIM)
    except ValueError:
        continue

    task = parts[1]
    attacker = parts[3]

    if len(parts) > victim_idx + 1:
        defense = parts[victim_idx + 1]
        param = parts[victim_idx + 2] if len(parts) > victim_idx + 2 else '0'
    else:
        defense = 'none'
        param = '0'

    score = parse_bodega_score(fname)
    if score is not None:
        key = get_defense_key(defense, param)
        bodega_scores[key].append(score)
        bodega_by_task[key][task].append(score)

# Collect clean accuracy per defense
accuracy_data = {}
for fname in sorted(ACCURACY_DIR.glob(f'*{VICTIM}*.txt')):
    with open(fname) as f:
        content = f.read()
    # Try to extract defense name and accuracy
    defense_match = re.search(r'Defense: (\S+)', content)
    param_match = re.search(r'Param: (\S+)', content)
    acc_match = re.search(r'Accuracy: ([\d.]+)', content)
    f1_match = re.search(r'F1: ([\d.]+)', content)

    if acc_match:
        defense = defense_match.group(1) if defense_match else 'none'
        param = param_match.group(1) if param_match else '0'
        key = get_defense_key(defense, param)
        accuracy_data[key] = {
            'accuracy': float(acc_match.group(1)),
            'f1': float(f1_match.group(1)) if f1_match else None
        }

# If no clean accuracy files found, try to parse from eval_defense_accuracy results
if not accuracy_data:
    for fname in sorted(ACCURACY_DIR.glob(f'*{VICTIM}*.txt')):
        try:
            with open(fname) as f:
                for line in f:
                    line = line.strip()
                    if 'accuracy' in line.lower() and ':' in line:
                        parts_l = line.split(':')
                        if len(parts_l) >= 2:
                            try:
                                val = float(parts_l[-1].strip().rstrip('%'))
                                print(f"  Found accuracy {val} in {fname.name}")
                            except ValueError:
                                pass
        except Exception:
            pass

# Print ranking
print(f"\n{'='*80}")
print(f"DEFENSE RANKING — {VICTIM} (accuracy-penalized BODEGA score)")
print(f"{'='*80}")
print(f"{'Defense':<25s} {'Avg BODEGA':>10s} {'Clean Acc':>10s} "
      f"{'Effective':>10s} {'N results':>10s}")
print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

rankings = []
for key in sorted(bodega_scores.keys()):
    scores = bodega_scores[key]
    avg_bodega = sum(scores) / len(scores)
    acc = accuracy_data.get(key, {}).get('accuracy')
    if acc is not None:
        effective = acc * (1 - avg_bodega)
    else:
        effective = 1.0 * (1 - avg_bodega)  # assume perfect accuracy if unknown
    rankings.append((key, avg_bodega, acc, effective, len(scores)))

# Sort by effective score (higher = better defense)
rankings.sort(key=lambda x: x[3], reverse=True)

for key, avg_b, acc, eff, n in rankings:
    acc_str = f'{acc:.4f}' if acc is not None else '?'
    print(f'{key:<25s} {avg_b:>10.4f} {acc_str:>10s} {eff:>10.4f} {n:>10d}')

# Print per-task breakdown for top 5
print(f"\n{'='*80}")
print("TOP 5 — Per-task BODEGA scores")
print(f"{'='*80}")
for key, avg_b, acc, eff, n in rankings[:5]:
    print(f"\n  {key} (effective={eff:.4f}):")
    for task in TASKS:
        task_scores = bodega_by_task[key].get(task, [])
        if task_scores:
            print(f"    {task}: avg={sum(task_scores)/len(task_scores):.4f} "
                  f"(n={len(task_scores)})")

# Recommend best defense for XARELLO
print(f"\n{'='*80}")
print("RECOMMENDATION for XARELLO experiments")
print(f"{'='*80}")
if rankings:
    # Best overall (excluding 'none')
    best_non_none = [r for r in rankings if r[0] != 'none']
    if best_non_none:
        print(f"  Best defense: {best_non_none[0][0]}")
        print(f"    Avg BODEGA: {best_non_none[0][1]:.4f}")
        print(f"    Clean accuracy: {best_non_none[0][2]}")
        print(f"    Effective score: {best_non_none[0][3]:.4f}")
