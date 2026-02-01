import argparse
import gc
import os
import pathlib
import sys
import time
import random
import numpy as np

import victims.surprise

import OpenAttack
import torch
from datasets import Dataset

from metrics.BODEGAScore import BODEGAScore
from utils.data_mappings import dataset_mapping, dataset_mapping_pairs, SEPARATOR_CHAR
from utils.no_ssl_verify import no_ssl_verify
from victims.surprise import VictimRoBERTa
from victims.transformer import VictimTransformer, readfromfile_generator, PRETRAINED_BERT, PRETRAINED_GEMMA_2B, \
    PRETRAINED_GEMMA_7B
from victims.bilstm import VictimBiLSTM
from victims.caching import VictimCache
from victims.unk_fix_wrapper import UNK_TEXT
from defenses.preprocessing import get_defense

# Attempt at determinism
random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

# Running variables
print("Preparing the environment...")

# Default values
task = 'PR2'
targeted = True
attack = 'BERTattack'
victim_model = 'GEMMA'
out_dir = None
defense_type = 'none'
defense_param = 0.0
defense_seed = 42
verbose = False
data_path = pathlib.Path.home() / 'data' / 'BODEGA' / task
model_path = pathlib.Path.home() / 'data' / 'BODEGA' / task / (victim_model + '-512.pth')

# Parse arguments - support both legacy positional args and new named args
parser = argparse.ArgumentParser(description='BODEGA attack evaluation with optional defenses')
parser.add_argument('--defense', type=str, default='none',
                    help='Defense type: none, spellcheck, char_masking, char_noise')
parser.add_argument('--defense_param', type=float, default=0.0,
                    help='Defense parameter (dropout_prob or noise_std)')
parser.add_argument('--defense_seed', type=int, default=42,
                    help='Random seed for defense (for reproducibility)')
parser.add_argument('--verbose', action='store_true',
                    help='Print defense modifications as they happen')

# Check if using legacy positional args or new named args
if len(sys.argv) >= 7 and not sys.argv[1].startswith('--'):
    # Legacy positional argument parsing
    task = sys.argv[1]
    targeted = (sys.argv[2].lower() == 'true')
    attack = sys.argv[3]
    victim_model = sys.argv[4]
    data_path = pathlib.Path(sys.argv[5])
    model_path = pathlib.Path(sys.argv[6])
    if len(sys.argv) >= 8 and not sys.argv[7].startswith('--'):
        out_dir = pathlib.Path(sys.argv[7])
        remaining_args = sys.argv[8:]
    else:
        remaining_args = sys.argv[7:]

    # Parse defense args after positional args
    if remaining_args:
        args, _ = parser.parse_known_args(remaining_args)
        defense_type = args.defense
        defense_param = args.defense_param
        defense_seed = args.defense_seed
        verbose = args.verbose

# Build output filename including defense info
defense_suffix = ''
if defense_type != 'none':
    defense_suffix = f'_{defense_type}'
    if defense_param > 0:
        defense_suffix += f'_{defense_param}'

using_TF = (attack in ['TextFooler', 'BAE'])
FILE_NAME = f'results_{task}_{targeted}_{attack}_{victim_model}{defense_suffix}.txt'
if out_dir and (out_dir / FILE_NAME).exists():
    print("Report found, exiting...")
    sys.exit()

# Prepare task data
with_pairs = (task == 'FC' or task == 'C19')

# Choose device
print("Setting up the device...")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
if using_TF:
    # Disable GPU usage by TF to avoid memory conflicts
    import tensorflow as tf

    tf.config.set_visible_devices(devices=[], device_type='GPU')

if torch.cuda.is_available():
    victim_device = torch.device("cuda")
    attacker_device = torch.device("cuda")
else:
    victim_device = torch.device("cpu")
    attacker_device = torch.device('cpu')

# Prepare victim
print("Loading up victim model...")
pretrained_model = None
if victim_model == 'BERT':
    pretrained_model = PRETRAINED_BERT
    base_victim = VictimTransformer(model_path, task, pretrained_model, False, victim_device)
elif victim_model == 'GEMMA':
    pretrained_model = PRETRAINED_GEMMA_2B
    base_victim = VictimTransformer(model_path, task, pretrained_model, True, victim_device)
elif victim_model == 'GEMMA7B':
    pretrained_model = PRETRAINED_GEMMA_7B
    base_victim = VictimTransformer(model_path, task, pretrained_model, True, victim_device)
elif victim_model == 'BiLSTM':
    pretrained_model = PRETRAINED_BERT
    base_victim = VictimBiLSTM(model_path, task, victim_device)
elif victim_model == 'surprise':
    pretrained_model = victims.surprise.pretrained_model
    base_victim = VictimRoBERTa(model_path, task, victim_device)

# Apply defense wrapper if specified
defended_victim = None
if defense_type != 'none':
    print(f"Applying {defense_type} defense (param={defense_param}, seed={defense_seed})...")
    defended_victim = get_defense(defense_type, base_victim, param=defense_param, seed=defense_seed, verbose=verbose)
    # Skip caching when defense is active - defense transforms input so cache would be invalid
    victim = defended_victim
else:
    victim = VictimCache(model_path, base_victim)

# Load data
print("Loading data...")
test_dataset = Dataset.from_generator(readfromfile_generator,
                                      gen_kwargs={'subset': 'attack', 'dir': data_path,
                                                  'pretrained_model': pretrained_model, 'trim_text': True,
                                                  'with_pairs': with_pairs})
if not with_pairs:
    dataset = test_dataset.map(function=dataset_mapping)
    dataset = dataset.remove_columns(["text"])
else:
    dataset = test_dataset.map(function=dataset_mapping_pairs)
    dataset = dataset.remove_columns(["text1", "text2"])

dataset = dataset.remove_columns(["fake"])
# dataset = dataset.select(range(10))

# Filter data
if targeted:
    dataset = [inst for inst in dataset if inst["y"] == 1 and victim.get_pred([inst["x"]])[0] == inst["y"]]

print("Subset size: " + str(len(dataset)))

# Prepare attack
# Order for PR: DeepWordBug TextFooler PWWS BERTattack PSO BAE SCPN Genetic
print("Setting up the attacker...")
filter_words = OpenAttack.attack_assist.filter_words.get_default_filter_words('english') + [SEPARATOR_CHAR]
# Necessary to bypass the outdated SSL certifiacte on the OpenAttack servers
with no_ssl_verify():
    if attack == 'PWWS':
        attacker = OpenAttack.attackers.PWWSAttacker(token_unk=UNK_TEXT, lang='english', filter_words=filter_words)
    elif attack == 'SCPN':
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        attacker = OpenAttack.attackers.SCPNAttacker(device=attacker_device)
    elif attack == 'TextFooler':
        attacker = OpenAttack.attackers.TextFoolerAttacker(token_unk=UNK_TEXT, lang='english',
                                                           filter_words=filter_words)
    elif attack == 'DeepWordBug':
        attacker = OpenAttack.attackers.DeepWordBugAttacker(token_unk=UNK_TEXT)
    elif attack == 'VIPER':
        attacker = OpenAttack.attackers.VIPERAttacker()
    elif attack == 'GAN':
        attacker = OpenAttack.attackers.GANAttacker()
    elif attack == 'Genetic':
        attacker = OpenAttack.attackers.GeneticAttacker(lang='english', filter_words=filter_words)
    elif attack == 'PSO':
        attacker = OpenAttack.attackers.PSOAttacker(lang='english', filter_words=filter_words)
    elif attack == 'BERTattack':
        attacker = OpenAttack.attackers.BERTAttacker(filter_words=filter_words, use_bpe=False, device=attacker_device)
    elif attack == 'BAE':
        attacker = OpenAttack.attackers.BAEAttacker(device=attacker_device, filter_words=filter_words)
    else:
        attacker = None

# Run the attack
print("Evaluating the attack...")
RAW_FILE_NAME = f'raw_{task}_{targeted}_{attack}_{victim_model}{defense_suffix}.tsv'
raw_path = None  # Don't save raw attack details (saves disk space)
with no_ssl_verify():
    # Use BERTscore instead of BLEURT (BLEURT causes mutex crash on macOS)
    scorer = BODEGAScore(victim_device, task, align_sentences=True, semantic_scorer="BERTscore", raw_path=raw_path)
    attack_eval = OpenAttack.AttackEval(attacker, victim, language='english', metrics=[
        scorer  # , OpenAttack.metric.EditDistance()
    ])
    start = time.time()
    summary = attack_eval.eval(dataset, visualize=True, progress_bar=False)
    end = time.time()
attack_time = end - start
attacker = None

# Save defense modifications if any
if defended_victim is not None and out_dir:
    modifications = defended_victim.get_modifications()
    if modifications:
        mod_file = out_dir / f'modifications_{task}_{targeted}_{attack}_{victim_model}{defense_suffix}.tsv'
        defended_victim.save_modifications(str(mod_file))
        print(f"Saved {len(modifications)} defense modifications to {mod_file}")

# Remove unused stuff
victim.finalise()
del victim
gc.collect()
torch.cuda.empty_cache()
if "TOKENIZERS_PARALLELISM" in os.environ:
    del os.environ["TOKENIZERS_PARALLELISM"]

# Evaluate
start = time.time()
score_success, score_semantic, score_character, score_BODEGA = scorer.compute()
end = time.time()
evaluate_time = end - start

# Print results
print("=" * 50)
print("EXPERIMENT CONFIGURATION")
print("=" * 50)
print(f"Task: {task}")
print(f"Targeted: {targeted}")
print(f"Attack: {attack}")
print(f"Victim: {victim_model}")
print(f"Defense: {defense_type}")
if defense_type != 'none':
    print(f"Defense param: {defense_param}")
    print(f"Defense seed: {defense_seed}")
print("=" * 50)
print("RESULTS")
print("=" * 50)
print("Subset size: " + str(len(dataset)))
print("Success score: " + str(score_success))
print("Semantic score: " + str(score_semantic))
print("Character score: " + str(score_character))
print("BODEGA score: " + str(score_BODEGA))
print("Queries per example: " + str(summary['Avg. Victim Model Queries']))
print("Total attack time: " + str(attack_time))
print("Time per example: " + str((attack_time) / len(dataset)))
print("Total evaluation time: " + str(evaluate_time))

if out_dir:
    with open(out_dir / FILE_NAME, 'w') as f:
        f.write("# Experiment Configuration\n")
        f.write(f"Task: {task}\n")
        f.write(f"Targeted: {targeted}\n")
        f.write(f"Attack: {attack}\n")
        f.write(f"Victim: {victim_model}\n")
        f.write(f"Defense: {defense_type}\n")
        f.write(f"Defense param: {defense_param}\n")
        f.write(f"Defense seed: {defense_seed}\n")
        f.write("\n# Results\n")
        f.write("Subset size: " + str(len(dataset)) + '\n')
        f.write("Success score: " + str(score_success) + '\n')
        f.write("Semantic score: " + str(score_semantic) + '\n')
        f.write("Character score: " + str(score_character) + '\n')
        f.write("BODEGA score: " + str(score_BODEGA) + '\n')
        f.write("Queries per example: " + str(summary['Avg. Victim Model Queries']) + '\n')
        f.write("Total attack time: " + str(attack_time) + '\n')
        f.write("Time per example: " + str((attack_time) / len(dataset)) + '\n')
        f.write("Total evaluation time: " + str(evaluate_time) + '\n')