#!/bin/bash
# Launch Ultra-Stable Training for Final Push to 70%
# This script uses the best model so far (64.64%) and applies ultra-stable training

set -e

echo "========================================================================"
echo "ULTRA-STABLE TRAINING - FINAL PUSH TO 70%"
echo "========================================================================"
echo ""
echo "Strategy:"
echo "  ✓ Large batch (64) for stable gradients"
echo "  ✓ Very low LR (0.0001) for minimal updates"
echo "  ✓ AdamW optimizer for better convergence"
echo "  ✓ Minimal augmentation to preserve learned features"
echo "  ✓ Aggressive early stopping (patience=50)"
echo ""
echo "Expected:"
echo "  - Smooth training curves (not jumpy)"
echo "  - Slower but steadier convergence"
echo "  - Target: 70%+ accuracy"
echo "  - Model size: < 75MB"
echo ""
echo "========================================================================"
echo ""

# Configuration
DATASET="/scratch/am14419/projects/cap_11/dataset_root"
BASE_MODEL="/scratch/am14419/projects/cap_11/runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt"
EXPERIMENT_NAME="ultra_stable_v1"
HYP_FILE="hyp_fish_ultra_stable.yaml"

# Validate paths
if [ ! -d "$DATASET" ]; then
    echo "❌ ERROR: Dataset not found at: $DATASET"
    exit 1
fi

if [ ! -f "$BASE_MODEL" ]; then
    echo "❌ ERROR: Base model not found at: $BASE_MODEL"
    exit 1
fi

if [ ! -f "$HYP_FILE" ]; then
    echo "❌ ERROR: Hyperparameters file not found: $HYP_FILE"
    exit 1
fi

echo "✓ Validated paths"
echo "  Dataset: $DATASET"
echo "  Base model: $BASE_MODEL"
echo "  Hyperparameters: $HYP_FILE"
echo ""

# Check GPU
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Start training
echo "🚀 Starting ultra-stable training..."
echo ""

python train_ultra_stable.py \
    --data "$DATASET" \
    --model "$BASE_MODEL" \
    --batch 64 \
    --epochs 300 \
    --patience 50 \
    --device 0 \
    --workers 4 \
    --hyp "$HYP_FILE" \
    --name "$EXPERIMENT_NAME" \
    2>&1 | tee training_ultra_stable_v1.log

echo ""
echo "========================================================================"
echo "✓ TRAINING COMPLETE"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Output: /scratch/am14419/projects/cap_11/runs/detect/$EXPERIMENT_NAME"
echo "  Log: training_ultra_stable_v1.log"
echo ""
echo "Next steps:"
echo "  1. Check accuracy: cat /scratch/am14419/projects/cap_11/runs/detect/$EXPERIMENT_NAME/metrics_summary.csv | tail -5"
echo "  2. Check model size: ls -lh /scratch/am14419/projects/cap_11/runs/detect/$EXPERIMENT_NAME/weights/best.pt"
echo "  3. Compare with baseline (64.64%)"
echo ""
echo "If accuracy < 70%, try:"
echo "  - Increase batch to 80 or 96"
echo "  - Reduce LR to 0.00005"
echo "  - Train longer (500 epochs with patience=100)"
echo ""
echo "========================================================================"
