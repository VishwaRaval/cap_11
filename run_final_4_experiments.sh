#!/bin/bash
# Run 4 final experiments: 2 precision-focused + 2 recall-focused

set -e  # Exit on error

# Configuration
DATASET_ROOT="/scratch/am14419/projects/cap_11/dataset_root"
BASE_MODEL="/scratch/am14419/projects/cap_11/runs/detect/extreme_stable_v1/weights/best.pt"
EPOCHS=300
BATCH=80        # Even larger batch for maximum stability
PATIENCE=60     # Longer patience
DEVICE="0"      # Change if needed

echo "=========================================="
echo "FINAL 4 EXPERIMENTS: PRECISION + RECALL"
echo "=========================================="
echo "Dataset: $DATASET_ROOT"
echo "Base Model: $BASE_MODEL"
echo "Batch Size: $BATCH"
echo "Max Epochs: $EPOCHS"
echo "Patience: $PATIENCE"
echo "=========================================="
echo ""

# Experiment 1: Precision Focus v1
echo "▶ [1/4] Precision Focus v1 (High confidence + Strong classification)"
python train_ultra_stable.py \
    --data "$DATASET_ROOT" \
    --model "$BASE_MODEL" \
    --epochs $EPOCHS \
    --batch $BATCH \
    --patience $PATIENCE \
    --device $DEVICE \
    --hyp hyp_precision_focus_v1.yaml \
    --name precision_focus_v1

echo ""
echo "✓ Experiment 1 complete!"
echo ""
sleep 5

# Experiment 2: Precision Focus v2
echo "▶ [2/4] Precision Focus v2 (Extreme classification + Ultra-conservative)"
python train_ultra_stable.py \
    --data "$DATASET_ROOT" \
    --model "$BASE_MODEL" \
    --epochs $EPOCHS \
    --batch $BATCH \
    --patience $PATIENCE \
    --device $DEVICE \
    --hyp hyp_precision_focus_v2.yaml \
    --name precision_focus_v2

echo ""
echo "✓ Experiment 2 complete!"
echo ""
sleep 5

# Experiment 3: Recall Focus v1
echo "▶ [3/4] Recall Focus v1 (Low confidence + Strong box detection)"
python train_ultra_stable.py \
    --data "$DATASET_ROOT" \
    --model "$BASE_MODEL" \
    --epochs $EPOCHS \
    --batch $BATCH \
    --patience $PATIENCE \
    --device $DEVICE \
    --hyp hyp_recall_focus_v1.yaml \
    --name recall_focus_v1

echo ""
echo "✓ Experiment 3 complete!"
echo ""
sleep 5

# Experiment 4: Recall Focus v2
echo "▶ [4/4] Recall Focus v2 (Extreme box focus + Balanced)"
python train_ultra_stable.py \
    --data "$DATASET_ROOT" \
    --model "$BASE_MODEL" \
    --epochs $EPOCHS \
    --batch $BATCH \
    --patience $PATIENCE \
    --device $DEVICE \
    --hyp hyp_recall_focus_v2.yaml \
    --name recall_focus_v2

echo ""
echo "✓ Experiment 4 complete!"
echo ""

# Summary
echo "=========================================="
echo "ALL 4 EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  /scratch/am14419/projects/cap_11/runs/detect/precision_focus_v1/"
echo "  /scratch/am14419/projects/cap_11/runs/detect/precision_focus_v2/"
echo "  /scratch/am14419/projects/cap_11/runs/detect/recall_focus_v1/"
echo "  /scratch/am14419/projects/cap_11/runs/detect/recall_focus_v2/"
echo ""
echo "Next steps:"
echo "  1. Compare all 4 results"
echo "  2. Check which achieves best precision/recall balance"
echo "  3. Potentially ensemble the best models"
echo "=========================================="
