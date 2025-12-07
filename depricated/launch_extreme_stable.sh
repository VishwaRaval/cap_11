#!/bin/bash
# Launch EXTREME Stability Training - Last Resort for 70%
# Use this if ultra_stable doesn't reach 70%

set -e

echo "========================================================================"
echo "EXTREME STABILITY TRAINING - MAXIMUM GRADIENT STABILITY"
echo "========================================================================"
echo ""
echo "⚠️  WARNING: This is the most conservative possible configuration"
echo ""
echo "Strategy:"
echo "  ✓ VERY large batch (80) for maximum gradient stability"
echo "  ✓ VERY low LR (0.00005) for minimal updates"
echo "  ✓ AdamW optimizer"
echo "  ✓ ZERO augmentation (only horizontal flip)"
echo "  ✓ Maximum regularization (dropout=0.30)"
echo "  ✓ Very long training (500 epochs, patience=100)"
echo ""
echo "Expected:"
echo "  - Extremely smooth training curves (almost flat)"
echo "  - Very slow convergence (may need 300+ epochs)"
echo "  - Maximum stability and generalization"
echo "  - Target: 69-73% accuracy"
echo "  - Training time: ~12-16 hours"
echo ""
echo "⚠️  Use this ONLY if ultra_stable_v1 fails to reach 70%"
echo ""
echo "========================================================================"
echo ""

# Configuration
DATASET="/scratch/am14419/projects/cap_11/dataset_root"
BASE_MODEL="/scratch/am14419/projects/cap_11/runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt"
EXPERIMENT_NAME="extreme_stable_v1"
HYP_FILE="hyp_fish_extreme_stable.yaml"

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
echo ""

# Check GPU memory (need ~40GB for batch 80)
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE_MEM" -lt 35000 ]; then
    echo "⚠️  WARNING: Low GPU memory (${FREE_MEM}MB free)"
    echo "   Batch 80 may need ~40GB. Consider reducing to batch 64."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Confirm extreme settings
echo "⚠️  CONFIRMATION REQUIRED"
echo ""
echo "This will train with EXTREME stability settings:"
echo "  - Batch size: 80 (only ~37 updates per epoch!)"
echo "  - Learning rate: 0.00005 (200x lower than default)"
echo "  - Training time: ~12-16 hours"
echo ""
read -p "Are you sure you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Consider using ultra_stable_v1 instead."
    exit 1
fi

# Start training
echo ""
echo "🚀 Starting EXTREME stability training..."
echo ""

python train_ultra_stable.py \
    --data "$DATASET" \
    --model "$BASE_MODEL" \
    --batch 80 \
    --epochs 500 \
    --patience 100 \
    --device 0 \
    --workers 4 \
    --hyp "$HYP_FILE" \
    --name "$EXPERIMENT_NAME" \
    2>&1 | tee training_extreme_stable_v1.log

echo ""
echo "========================================================================"
echo "✓ EXTREME STABILITY TRAINING COMPLETE"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Output: /scratch/am14419/projects/cap_11/runs/detect/$EXPERIMENT_NAME"
echo "  Log: training_extreme_stable_v1.log"
echo ""
echo "Next steps:"
echo "  1. Check accuracy: tail -5 /scratch/am14419/projects/cap_11/runs/detect/$EXPERIMENT_NAME/metrics_summary.csv"
echo "  2. Verify < 75MB: ls -lh /scratch/am14419/projects/cap_11/runs/detect/$EXPERIMENT_NAME/weights/best.pt"
echo ""
echo "If STILL < 70%, consider:"
echo "  - Ensemble with top 3 models"
echo "  - Try YOLOv11m (if < 75MB after quantization)"
echo ""
echo "========================================================================"
