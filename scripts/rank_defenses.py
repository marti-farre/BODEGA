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
ACCURACY_DIR = Path('results/experiment-7_bleurt')  # clean_accuracy files are here too

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


def normalize_param(param):
    """Normalize param to consistent string: '-' and '0' -> '0', '7.00' -> '7.0'."""
    if param in ('-', '', None):
        return '0'
    try:
        return str(float(param))
    except ValueError:
        return param


def get_defense_key(defense, param):
    """Build human-readable defense key."""
    if defense == 'none':
        return 'none'
    param = normalize_param(param)
    if param in ('0.0', '0'):
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
# Files are named: clean_accuracy_TASK_VICTIM.txt
# They may contain per-defense lines or single defense results
accuracy_per_task = defaultdict(lambda: defaultdict(list))  # defense_key -> task -> [acc]
f1_per_task = defaultdict(lambda: defaultdict(list))  # defense_key -> task -> [f1]
accuracy_data = {}

for fname in sorted(ACCURACY_DIR.glob(f'clean_accuracy_*{VICTIM}*.txt')):
    task_from_fname = fname.stem.replace('clean_accuracy_', '').replace(f'_{VICTIM}', '')
    with open(fname) as f:
        content = f.read()

    # Format 1: Single defense per file with "Defense: X", "Accuracy: X"
    defense_match = re.search(r'Defense: (\S+)', content)
    param_match = re.search(r'Param: (\S+)', content)
    acc_match = re.search(r'Accuracy: ([\d.]+)', content)
    f1_match = re.search(r'F1: ([\d.]+)', content)

    if acc_match:
        defense = defense_match.group(1) if defense_match else 'none'
        param = param_match.group(1) if param_match else '0'
        key = get_defense_key(defense, param)
        acc_val = float(acc_match.group(1))
        accuracy_per_task[key][task_from_fname].append(acc_val)

    # Format 2: Multi-defense table (defense, param, accuracy, f1 per line)
    for line in content.splitlines():
        m = re.match(r'(\S+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)', line)
        if m:
            defense, param = m.group(1), m.group(2)
            acc_val, f1_val = float(m.group(3)), float(m.group(4))
            key = get_defense_key(defense, param)
            accuracy_per_task[key][task_from_fname].append(acc_val)
            f1_per_task[key][task_from_fname].append(f1_val)

# Average accuracy and F1 across tasks for each defense
f1_data = {}
for key, task_accs in accuracy_per_task.items():
    all_accs = [a for accs in task_accs.values() for a in accs]
    if all_accs:
        accuracy_data[key] = {
            'accuracy': sum(all_accs) / len(all_accs),
            'per_task': {t: sum(a)/len(a) for t, a in task_accs.items()}
        }
for key, task_f1s in f1_per_task.items():
    all_f1s = [f for f1s in task_f1s.values() for f in f1s]
    if all_f1s:
        f1_data[key] = sum(all_f1s) / len(all_f1s)

# Print ranking
print(f"\n{'='*80}")
print(f"DEFENSE RANKING — {VICTIM} (F1-penalized BODEGA score)")
print(f"{'='*80}")
print(f"{'Defense':<25s} {'Avg BODEGA':>10s} {'Clean Acc':>10s} {'Clean F1':>10s} "
      f"{'Eff(Acc)':>10s} {'Eff(F1)':>10s} {'N results':>10s}")
print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

rankings = []
for key in sorted(bodega_scores.keys()):
    scores = bodega_scores[key]
    avg_bodega = sum(scores) / len(scores)
    acc = accuracy_data.get(key, {}).get('accuracy')
    f1 = f1_data.get(key)
    eff_acc = (acc if acc is not None else 1.0) * (1 - avg_bodega)
    eff_f1 = (f1 if f1 is not None else 1.0) * (1 - avg_bodega)
    rankings.append((key, avg_bodega, acc, f1, eff_acc, eff_f1, len(scores)))

# Sort by F1-based effective score (higher = better defense)
rankings.sort(key=lambda x: x[5], reverse=True)

for key, avg_b, acc, f1, eff_a, eff_f, n in rankings:
    acc_str = f'{acc:.4f}' if acc is not None else '?'
    f1_str = f'{f1:.4f}' if f1 is not None else '?'
    print(f'{key:<25s} {avg_b:>10.4f} {acc_str:>10s} {f1_str:>10s} {eff_a:>10.4f} {eff_f:>10.4f} {n:>10d}')

# Print per-task breakdown for top 5
print(f"\n{'='*80}")
print("TOP 5 — Per-task BODEGA scores")
print(f"{'='*80}")
for key, avg_b, acc, f1, eff_a, eff_f, n in rankings[:5]:
    print(f"\n  {key} (eff_acc={eff_a:.4f}, eff_f1={eff_f:.4f}):")
    for task in TASKS:
        task_scores = bodega_by_task[key].get(task, [])
        if task_scores:
            task_acc = accuracy_data.get(key, {}).get('per_task', {}).get(task)
            acc_info = f", clean_acc={task_acc:.4f}" if task_acc else ""
            print(f"    {task}: avg_bodega={sum(task_scores)/len(task_scores):.4f} "
                  f"(n={len(task_scores)}){acc_info}")

# Recommend best defense for XARELLO
print(f"\n{'='*80}")
print("RECOMMENDATION for XARELLO experiments")
print(f"{'='*80}")
if rankings:
    best_non_none = [r for r in rankings if r[0] != 'none']
    if best_non_none:
        r = best_non_none[0]
        print(f"  Best defense: {r[0]}")
        print(f"    Avg BODEGA: {r[1]:.4f}")
        print(f"    Clean accuracy: {r[2]}")
        print(f"    Clean F1: {r[3]}")
        print(f"    Effective (acc): {r[4]:.4f}")
        print(f"    Effective (F1):  {r[5]:.4f}")
