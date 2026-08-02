#!/bin/bash
# Quick test: Run a single fast experiment with BLEURT to verify it works.
# Run on HPC with: cd ~/BODEGA && sbatch scripts/test_bleurt.sh
# Or interactively: salloc -p short --gres=gpu:1 --mem=32G -c 4
#                   then: cd ~/BODEGA && bash scripts/test_bleurt.sh

#SBATCH -J test_bleurt
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH -o logs/test_bleurt_%j.out
#SBATCH -e logs/test_bleurt_%j.err

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."
mkdir -p results/test_bleurt logs

echo "=== Testing BLEURT scorer ==="
echo "Step 1: Testing BLEURT import..."
python -c "
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print('Loading BLEURT model...')
model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
print('BLEURT loaded successfully!')
# Quick test
refs = ['The cat sat on the mat.']
cands = ['The cat was sitting on the mat.']
inputs = tokenizer(refs, cands, padding='longest', return_tensors='pt').to(device)
with torch.no_grad():
    score = model(**inputs).logits.flatten().tolist()
print(f'Test score: {score[0]:.4f} (should be close to 1.0)')
print('BLEURT import test PASSED')
"

if [ $? -ne 0 ]; then
    echo "BLEURT import test FAILED — fix before launching full experiment"
    exit 1
fi

echo ""
echo "Step 2: Running a quick attack with BLEURT (DeepWordBug on PR2/BiLSTM, fastest combo)..."
python runs/attack.py PR2 false DeepWordBug BiLSTM \
    data/PR2 data/PR2/BiLSTM-512.pth results/test_bleurt \
    --defense none --semantic_scorer BLEURT

echo ""
echo "Step 3: Checking output..."
if [ -f results/test_bleurt/results_PR2_False_DeepWordBug_BiLSTM.txt ]; then
    echo "=== Result file ==="
    cat results/test_bleurt/results_PR2_False_DeepWordBug_BiLSTM.txt
    echo ""
    echo "=== BLEURT test PASSED ==="
else
    echo "=== BLEURT test FAILED — no output file ==="
    exit 1
fi
