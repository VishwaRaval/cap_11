#!/bin/bash
# Experiment 1: LAB Corrected + Conservative Training (Batch 32)
# Expected: 68-71% accuracy with stable validation loss

set -e  # Exit on error

EXPERIMENT_NAME="exp1_lab_conservative_b32"
ORIGINAL_DATASET="dataset_root"
BASE_MODEL="runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt"
OUTPUT_DIR="runs/detect/${EXPERIMENT_NAME}"

echo "========================================================================"
echo "EXPERIMENT 1: LAB + Conservative Training (Batch 32)"
echo "========================================================================"
echo "Strategy: Image quality fix + prevent overfitting"
echo "Expected: 68-71% accuracy"
echo ""
echo "Key settings:"
echo "  - LAB preprocessing: blue_reduction=0.7"
echo "  - Learning rate: 0.0005 (20x lower)"
echo "  - Batch size: 32 (50% fewer updates)"
echo "  - Epochs: 200 with patience 100"
echo "  - Optimizer: AdamW"
echo "========================================================================"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Step 1: LAB Preprocessing
echo "[Step 1/3] Running LAB preprocessing..."
python preprocess_lab_underwater.py \
    --input ${ORIGINAL_DATASET} \
    --output ${OUTPUT_DIR}/dataset_corrected \
    --blue-reduction 0.7 \
    --gamma 1.0 \
    --visualize 10 \
    --quality-threshold 50 \
    2>&1 | tee ${OUTPUT_DIR}/preprocessing.log

echo ""
echo "✓ LAB preprocessing complete"
echo ""

# Step 2: Validate Preprocessing
echo "[Step 2/3] Validating preprocessing quality..."
python validate_lab_preprocessing.py \
    --original ${ORIGINAL_DATASET} \
    --corrected ${OUTPUT_DIR}/dataset_corrected \
    --output ${OUTPUT_DIR}/validation_report.png \
    2>&1 | tee ${OUTPUT_DIR}/validation.log

echo ""
echo "✓ Validation complete"
echo ""

# Step 3: Train Model
echo "[Step 3/3] Training YOLOv11 with conservative settings..."
python train_small_dataset.py \
    --data ${OUTPUT_DIR}/dataset_corrected \
    --model ${BASE_MODEL} \
    --batch 32 \
    --epochs 200 \
    --patience 100 \
    --imgsz 768 \
    --device 0 \
    --workers 4 \
    --hyp hyp_fish_small_dataset.yaml \
    --project ${OUTPUT_DIR}/runs \
    --name training \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "========================================================================"
echo "✓ EXPERIMENT 1 COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - Preprocessed dataset: ${OUTPUT_DIR}/dataset_corrected"
echo "  - Training results: ${OUTPUT_DIR}/runs/training"
echo "  - Best model: ${OUTPUT_DIR}/runs/training/weights/best.pt"
echo ""
echo "Key files:"
echo "  - preprocessing.log"
echo "  - validation.log"
echo "  - training.log"
echo "  - validation_report.png"
echo ""
echo "Next: Review results and compare with baseline (64.64%)"
echo "========================================================================"
