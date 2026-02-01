"""
Evaluate classifier accuracy on clean (unattacked) data with various defenses.

This script measures the utility-robustness trade-off by comparing:
- Baseline accuracy (no defense)
- Accuracy with each defense configuration

Usage:
    python runs/eval_defense_accuracy.py <task> <victim> <data_path> <model_path> [output_dir]

Example:
    python runs/eval_defense_accuracy.py PR2 BiLSTM data/PR2 data/PR2/BiLSTM-512.pth results
"""

import argparse
import pathlib
import sys
import torch
import numpy as np
from datasets import Dataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from victims.bilstm import VictimBiLSTM
from victims.transformer import VictimTransformer, readfromfile_generator, PRETRAINED_BERT, PRETRAINED_GEMMA_2B, PRETRAINED_GEMMA_7B
from defenses.preprocessing import get_defense
from utils.data_mappings import dataset_mapping, dataset_mapping_pairs

# Defense configurations to test
DEFENSE_CONFIGS = [
    ('none', 0.0),
    ('spellcheck', 0.0),
    ('char_dropout', 0.05),
    ('char_dropout', 0.10),
    ('char_dropout', 0.15),
    ('char_dropout', 0.20),
    ('char_noise', 0.05),
    ('char_noise', 0.10),
    ('char_noise', 0.15),
    ('char_noise', 0.20),
]

DEFENSE_SEED = 42


def evaluate_accuracy(victim, texts, labels, batch_size=32):
    """
    Evaluate accuracy of a victim classifier on given texts and labels.

    Args:
        victim: OpenAttack classifier (possibly wrapped with defense)
        texts: List of input texts
        labels: numpy array of true labels
        batch_size: Batch size for prediction

    Returns:
        accuracy: float
        predictions: numpy array
    """
    all_preds = []

    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating", leave=False):
        batch_texts = texts[i:i + batch_size]
        batch_preds = victim.get_pred(batch_texts)
        all_preds.extend(batch_preds.tolist())

    predictions = np.array(all_preds)
    accuracy = np.mean(predictions == labels)

    return accuracy, predictions


def compute_f1(predictions, labels):
    """Compute F1 score for binary classification."""
    TPs = np.sum((labels == 1) & (predictions == 1))
    FPs = np.sum((labels == 0) & (predictions == 1))
    FNs = np.sum((labels == 1) & (predictions == 0))

    if 2 * TPs + FPs + FNs == 0:
        return 0.0
    return 2 * TPs / (2 * TPs + FPs + FNs)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate defense accuracy on clean data')
    parser.add_argument('task', type=str, help='Task name (PR2, FC, C19, HN, RD)')
    parser.add_argument('victim', type=str, help='Victim model (BiLSTM, BERT, GEMMA)')
    parser.add_argument('data_path', type=str, help='Path to data directory')
    parser.add_argument('model_path', type=str, help='Path to model weights')
    parser.add_argument('output_dir', type=str, nargs='?', default=None, help='Output directory for results')
    parser.add_argument('--subset', type=str, default='attack', help='Data subset to use (attack or train)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for defenses')

    args = parser.parse_args()

    task = args.task
    victim_model = args.victim
    data_path = pathlib.Path(args.data_path)
    model_path = pathlib.Path(args.model_path)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    subset = args.subset
    defense_seed = args.seed

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine if task uses text pairs
    with_pairs = (task == 'FC' or task == 'C19')

    # Load victim model
    print(f"Loading {victim_model} model...")
    if victim_model == 'BiLSTM':
        pretrained_model = PRETRAINED_BERT
        base_victim = VictimBiLSTM(model_path, task, device)
    elif victim_model == 'BERT':
        pretrained_model = PRETRAINED_BERT
        base_victim = VictimTransformer(model_path, task, pretrained_model, False, device)
    elif victim_model == 'GEMMA':
        pretrained_model = PRETRAINED_GEMMA_2B
        base_victim = VictimTransformer(model_path, task, pretrained_model, True, device)
    elif victim_model == 'GEMMA7B':
        pretrained_model = PRETRAINED_GEMMA_7B
        base_victim = VictimTransformer(model_path, task, pretrained_model, True, device)
    else:
        raise ValueError(f"Unknown victim model: {victim_model}")

    # Load test data
    print(f"Loading {subset} data from {data_path}...")
    test_dataset = Dataset.from_generator(
        readfromfile_generator,
        gen_kwargs={
            'subset': subset,
            'dir': data_path,
            'pretrained_model': pretrained_model,
            'trim_text': True,
            'with_pairs': with_pairs
        }
    )

    # Map to standard format
    if not with_pairs:
        dataset = test_dataset.map(function=dataset_mapping)
    else:
        dataset = test_dataset.map(function=dataset_mapping_pairs)

    # Extract texts and labels
    texts = [item['x'] for item in dataset]
    labels = np.array([item['y'] for item in dataset])

    print(f"Loaded {len(texts)} examples")
    print(f"Label distribution: {np.sum(labels == 0)} negative, {np.sum(labels == 1)} positive")

    # Results storage
    results = []
    baseline_accuracy = None

    # Evaluate each defense configuration
    print("\n" + "=" * 60)
    print(f"Clean Accuracy Evaluation ({task}, {victim_model})")
    print("=" * 60)

    for defense_name, defense_param in DEFENSE_CONFIGS:
        # Create defended victim
        if defense_name == 'none':
            victim = base_victim
            config_str = "baseline (no defense)"
        else:
            victim = get_defense(defense_name, base_victim, param=defense_param, seed=defense_seed)
            if defense_param > 0:
                config_str = f"{defense_name} (param={defense_param})"
            else:
                config_str = f"{defense_name}"

        print(f"\nEvaluating: {config_str}")

        # Evaluate
        accuracy, predictions = evaluate_accuracy(victim, texts, labels)
        f1 = compute_f1(predictions, labels)

        # Store results
        result = {
            'defense': defense_name,
            'param': defense_param,
            'accuracy': accuracy,
            'f1': f1,
        }

        if defense_name == 'none':
            baseline_accuracy = accuracy
            result['delta'] = 0.0
            result['delta_pct'] = 0.0
        else:
            result['delta'] = accuracy - baseline_accuracy
            result['delta_pct'] = (accuracy - baseline_accuracy) / baseline_accuracy * 100

        results.append(result)

        # Print result
        if defense_name == 'none':
            print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        else:
            print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Delta: {result['delta']:+.4f} ({result['delta_pct']:+.2f}%)")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Defense':<15} {'Param':<8} {'Accuracy':<10} {'F1':<10} {'Delta':<12}")
    print("-" * 60)

    for r in results:
        param_str = f"{r['param']:.2f}" if r['param'] > 0 else "-"
        delta_str = f"{r['delta']:+.4f} ({r['delta_pct']:+.2f}%)" if r['defense'] != 'none' else "-"
        print(f"{r['defense']:<15} {param_str:<8} {r['accuracy']:<10.4f} {r['f1']:<10.4f} {delta_str:<12}")

    # Save results if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"clean_accuracy_{task}_{victim_model}.txt"

        with open(output_file, 'w') as f:
            f.write(f"# Clean Accuracy Evaluation Results\n")
            f.write(f"Task: {task}\n")
            f.write(f"Victim: {victim_model}\n")
            f.write(f"Data subset: {subset}\n")
            f.write(f"Defense seed: {defense_seed}\n")
            f.write(f"Total examples: {len(texts)}\n")
            f.write(f"\n# Results\n")
            f.write(f"{'Defense':<15} {'Param':<8} {'Accuracy':<10} {'F1':<10} {'Delta':<12}\n")
            f.write("-" * 60 + "\n")

            for r in results:
                param_str = f"{r['param']:.2f}" if r['param'] > 0 else "-"
                delta_str = f"{r['delta']:+.4f} ({r['delta_pct']:+.2f}%)" if r['defense'] != 'none' else "-"
                f.write(f"{r['defense']:<15} {param_str:<8} {r['accuracy']:<10.4f} {r['f1']:<10.4f} {delta_str:<12}\n")

        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
