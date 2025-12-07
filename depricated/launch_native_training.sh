#!/bin/bash
# Launch Native Resolution Training
# This preserves the 768×432 aspect ratio without stretching

set -e

echo "========================================================================"
echo "NATIVE RESOLUTION TRAINING - NO STRETCHING"
echo "========================================================================"
echo ""
echo "Original images:  768×432 (16:9 aspect ratio)"
echo "Training size:    768×432 (NATIVE - no padding or stretching!)"
echo ""
echo "Advantages:"
echo "  ✅ No image distortion"
echo "  ✅ No unnecessary padding"
echo "  ✅ Better feature preservation"
echo "  ✅ Potential +0.5-2% accuracy improvement"
echo ""
echo "========================================================================"
echo ""

# Configuration
DATASET="/scratch/am14419/projects/cap_11/dataset_root"
BASE_MODEL="/scratch/am14419/projects/cap_11/runs/detect/extreme_stable_v1/weights/best.pt"
EXPERIMENT_NAME="extreme_stable_v2_native"
HYP_FILE="hyp_fish_moderate.yaml"

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

# Check GPU
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Confirmation
echo "⚠️  IMPORTANT CONFIRMATION"
echo ""
echo "This will train at NATIVE 768×432 resolution."
echo "Your existing models were trained at 768×768 (square)."
echo ""
echo "If you ensemble this with existing models, you should:"
echo "  1. Retrain ALL models at 768×432 for consistency, OR"
echo "  2. Use this as a standalone model"
echo ""
read -p "Continue with native resolution training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "🚀 Starting NATIVE resolution training..."
echo ""

python train_native_resolution.py \
    --data "$DATASET" \
    --model "$BASE_MODEL" \
    --batch 64 \
    --epochs 150 \
    --patience 50 \
    --device 0 \
    --workers 4 \
    --hyp "$HYP_FILE" \
    --name "$EXPERIMENT_NAME" \
    --imgsz 768 432 \
    2>&1 | tee training_${EXPERIMENT_NAME}.log

echo ""
echo "========================================================================"
echo "✓ TRAINING COMPLETE"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Output: /scratch/am14419/projects/cap_11/runs/detect/$EXPERIMENT_NAME"
echo "  Log: training_${EXPERIMENT_NAME}.log"
echo ""
echo "Next steps:"
echo "  1. Check accuracy vs 768×768 models"
echo "  2. If better: Consider retraining all models at native resolution"
echo "  3. If similar: Ensemble improvements are from better hyperparams, not resolution"
echo ""
echo "========================================================================"
