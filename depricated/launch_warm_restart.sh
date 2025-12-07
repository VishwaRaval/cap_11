#!/bin/bash
# Warm Restart from Ultra-Stable V1 Best Checkpoint
# V1 reached 66.59% at epoch 18 - continue training from there!

set -e

echo "========================================================================"
echo "WARM RESTART FROM ULTRA-STABLE V1"
echo "========================================================================"
echo ""
echo "🎯 Strategy:"
echo "  ✓ Start from v1's best checkpoint (66.59% at epoch 18)"
echo "  ✓ Slightly higher LR (0.00015) to push past plateau"
echo "  ✓ Batch 72 (middle ground between 64 and 80)"
echo "  ✓ Train 200 more epochs with patience=75"
echo ""
echo "📊 V1 Performance:"
echo "  - Best: 66.59% at epoch 18"
echo "  - Gap to target: 3.41%"
echo "  - Training was stable and improving"
echo ""
echo "Expected:"
echo "  - 68-71% accuracy (gentle push from 66.59%)"
echo "  - ~5-6 hours training time"
echo ""
echo "========================================================================"
echo ""

# Configuration
DATASET="/scratch/am14419/projects/cap_11/dataset_root"
BASE_MODEL="/scratch/am14419/projects/cap_11/runs/detect/ultra_stable_v1/weights/best.pt"
EXPERIMENT_NAME="ultra_stable_v1_restart"
HYP_FILE="hyp_fish_warm_restart.yaml"

# Create warm restart hyperparameters
cat > "$HYP_FILE" << 'EOF'
# Warm Restart Hyperparameters
# Slightly more aggressive than ultra_stable to push past 66.59%

# Learning rate - SLIGHTLY HIGHER for continued learning
lr0: 0.00015              # Was 0.0001 - increase by 50%
lrf: 0.000015             # 10% of initial
momentum: 0.95            
weight_decay: 0.0015      
warmup_epochs: 10.0       # Shorter warmup (already trained)
warmup_momentum: 0.85     
warmup_bias_lr: 0.00005   

# Loss weights - BALANCED
box: 7.5
cls: 1.5                  
dfl: 1.5

# Augmentation - MINIMAL
degrees: 0.5              
translate: 0.02           
scale: 0.1                
shear: 0.0                
perspective: 0.0          
flipud: 0.0               
fliplr: 0.5               

# Photometric - MINIMAL
hsv_h: 0.001              
hsv_s: 0.05               
hsv_v: 0.03               

# Multi-class augmentations - DISABLED
mosaic: 0.0               
mixup: 0.0                
copy_paste: 0.0           

# NMS settings
iou: 0.5
conf: 0.25

# Training stability
close_mosaic: 0           
dropout: 0.22             # Slightly less than before
label_smoothing: 0.12     # Slightly less than before
EOF

echo "✓ Created warm restart hyperparameters: $HYP_FILE"
echo ""

# Validate paths
if [ ! -d "$DATASET" ]; then
    echo "❌ ERROR: Dataset not found at: $DATASET"
    exit 1
fi

if [ ! -f "$BASE_MODEL" ]; then
    echo "❌ ERROR: V1 best model not found at: $BASE_MODEL"
    echo "   Make sure ultra_stable_v1 has completed training"
    exit 1
fi

echo "✓ Validated paths"
echo ""

# Check GPU
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Start warm restart training
echo "🚀 Starting warm restart from 66.59%..."
echo ""

python train_ultra_stable.py \
    --data "$DATASET" \
    --model "$BASE_MODEL" \
    --batch 72 \
    --epochs 200 \
    --patience 75 \
    --device 0 \
    --workers 4 \
    --hyp "$HYP_FILE" \
    --name "$EXPERIMENT_NAME" \
    2>&1 | tee training_warm_restart.log

echo ""
echo "========================================================================"
echo "✓ WARM RESTART COMPLETE"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Output: /scratch/am14419/projects/cap_11/runs/detect/$EXPERIMENT_NAME"
echo "  Log: training_warm_restart.log"
echo ""
echo "Compare with v1:"
echo "  V1 best: 66.59%"
echo "  V1 restart: Check metrics_summary.csv"
echo ""
echo "========================================================================"
