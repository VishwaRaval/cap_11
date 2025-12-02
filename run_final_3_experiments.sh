#!/bin/bash
# Final 3 Experiments to Break 70% Accuracy Barrier
# Estimated total time: ~2.5-3 hours on A100

set -e  # Exit on error

echo "========================================================================"
echo "FINAL 3 EXPERIMENTS - TARGETING >70% ACCURACY"
echo "========================================================================"
echo "Strategy:"
echo "  1. YOLOv11m (larger model) with recall optimization"
echo "  2. YOLOv11s with aggressive recall tuning + longer training"
echo "  3. YOLOv11s with class-balanced loss + focal loss simulation"
echo "========================================================================"
echo ""

# ============================================================================
# EXPERIMENT 1: YOLOv11m with Recall Optimization (~50-60 min)
# ============================================================================
# Rationale: Your best s model got 61.48%. A larger model (m) with more 
# capacity might break through 70% with the same recall-optimized strategy
# ============================================================================

echo ""
echo "========================================================================"
echo "EXPERIMENT 1: YOLOv11m Recall-Optimized"
echo "========================================================================"
echo "Model: yolo11m (~20M params vs 11M for s)"
echo "Strategy: Same recall optimization that worked for s model"
echo "Expected: ~65-72% accuracy (larger model capacity)"
echo "Time: ~50-60 minutes"
echo "========================================================================"
echo ""

python train_yolo11_fish_enhanced_fixed.py \
    --data dataset_root \
    --model m \
    --epochs 120 \
    --batch 8 \
    --early-stop-patience 40 \
    --use-class-weights \
    --name m_recall_optimized_v1

echo "✓ Experiment 1 complete!"
echo ""

# ============================================================================
# EXPERIMENT 2: YOLOv11s Aggressive Recall + Extended Training (~45-50 min)
# ============================================================================
# Rationale: Your s model plateaued. Try more aggressive hyperparameters
# with extended training to push recall higher
# ============================================================================

echo ""
echo "========================================================================"
echo "EXPERIMENT 2: YOLOv11s Aggressive Recall Tuning"
echo "========================================================================"
echo "Model: yolo11s"
echo "Strategy: Higher cls loss (1.5), lower conf threshold (0.15)"
echo "Expected: ~62-70% accuracy (aggressive recall focus)"
echo "Time: ~45-50 minutes"
echo "========================================================================"
echo ""

# Create aggressive hyperparameters
cat > hyp_fish_aggressive.yaml << 'EOF'
# Aggressive Recall Hyperparameters
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# AGGRESSIVE class loss for recall
box: 7.5
cls: 1.5  # Increased from 1.3
dfl: 1.5

# Conservative augmentation (keep what works)
degrees: 3.0
translate: 0.1
scale: 0.4
shear: 2.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5

hsv_h: 0.01
hsv_s: 0.3
hsv_v: 0.2

mosaic: 1.0
mixup: 0.05
copy_paste: 0.0

# LOWER thresholds for more detections (recall boost)
iou: 0.5
conf: 0.15  # Lowered from 0.25

close_mosaic: 10
dropout: 0.0
label_smoothing: 0.0
EOF

python train_yolo11_fish_enhanced_fixed.py \
    --data dataset_root \
    --model s \
    --epochs 150 \
    --batch 12 \
    --early-stop-patience 50 \
    --use-class-weights \
    --name s_aggressive_recall_v1

echo "✓ Experiment 2 complete!"
echo ""

# ============================================================================
# EXPERIMENT 3: YOLOv11s Multi-Scale + Heavy Augmentation (~45-50 min)
# ============================================================================
# Rationale: Fish vary in size. Multi-scale training + heavier augmentation
# might improve generalization and recall on small/partial fish
# ============================================================================

echo ""
echo "========================================================================"
echo "EXPERIMENT 3: YOLOv11s Multi-Scale + Heavy Aug"
echo "========================================================================"
echo "Model: yolo11s"
echo "Strategy: Multi-scale (640-896), heavier augmentation for robustness"
echo "Expected: ~63-71% accuracy (better generalization)"
echo "Time: ~45-50 minutes"
echo "========================================================================"
echo ""

# Create multi-scale hyperparameters
cat > hyp_fish_multiscale.yaml << 'EOF'
# Multi-Scale + Heavy Augmentation
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

box: 7.5
cls: 1.4
dfl: 1.5

# HEAVIER augmentation for robustness
degrees: 5.0      # Increased
translate: 0.15   # Increased
scale: 0.6        # Increased range
shear: 3.0        # Increased
perspective: 0.0
flipud: 0.0
fliplr: 0.5

hsv_h: 0.02       # Increased
hsv_s: 0.4        # Increased
hsv_v: 0.3        # Increased

mosaic: 1.0
mixup: 0.1        # Increased
copy_paste: 0.05  # Added light copy-paste

iou: 0.5
conf: 0.20        # Balanced

close_mosaic: 10
dropout: 0.0
label_smoothing: 0.0
EOF

# Create multi-scale training script
cat > train_multiscale.py << 'EOFPY'
#!/usr/bin/env python3
import os
from ultralytics import YOLO

# Set W&B
os.environ.setdefault('WANDB_PROJECT', 'underwater-fish-yolo11')

model = YOLO('yolo11s.pt')

results = model.train(
    data='dataset_root/data.yaml',
    epochs=150,
    batch=12,
    imgsz=[640, 768, 896],  # Multi-scale training
    device='0',
    workers=4,
    project='runs/detect',
    name='fish_s_multiscale_heavy_aug_v1',
    exist_ok=False,
    pretrained=True,
    optimizer='auto',
    verbose=True,
    seed=42,
    deterministic=True,
    single_cls=False,
    rect=False,  # Important: must be False for multi-scale
    cos_lr=False,
    close_mosaic=10,
    resume=False,
    amp=True,
    patience=50,
    plots=True,
    val=True,
    cfg='hyp_fish_multiscale.yaml',
)
EOFPY

chmod +x train_multiscale.py
python train_multiscale.py

echo "✓ Experiment 3 complete!"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================================================"
echo "ALL 3 EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in: runs/detect/"
echo ""
echo "Next steps:"
echo "  1. Evaluate all 3 models:"
echo "     python evaluate_all_models.py"
echo ""
echo "  2. Check W&B dashboard for detailed metrics"
echo ""
echo "  3. Best model likely in:"
echo "     - fish_m_recall_optimized_v1  (if capacity was the issue)"
echo "     - fish_s_aggressive_recall_v1 (if recall tuning helps)"
echo "     - fish_s_multiscale_heavy_aug_v1 (if generalization helps)"
echo "========================================================================"
