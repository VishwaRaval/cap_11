#!/bin/bash
# Experiment 2: LAB Corrected + Ultra-Conservative Training (Batch 48)
# Expected: 69-72% accuracy with very stable validation loss

set -e  # Exit on error

EXPERIMENT_NAME="exp2_lab_ultraconservative_b48"
ORIGINAL_DATASET="dataset_root"
BASE_MODEL="runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt"
OUTPUT_DIR="runs/detect/${EXPERIMENT_NAME}"

echo "========================================================================"
echo "EXPERIMENT 2: LAB + Ultra-Conservative Training (Batch 48)"
echo "========================================================================"
echo "Strategy: Maximum overfitting prevention with larger batches"
echo "Expected: 69-72% accuracy"
echo ""
echo "Key settings:"
echo "  - LAB preprocessing: blue_reduction=0.7"
echo "  - Learning rate: 0.0005 (20x lower)"
echo "  - Batch size: 48 (67% fewer updates than baseline)"
echo "  - Epochs: 250 with patience 150"
echo "  - Optimizer: AdamW"
echo "  - Even stronger regularization"
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

# Step 3: Train Model with Larger Batch
echo "[Step 3/3] Training YOLOv11 with ultra-conservative settings..."
python train_small_dataset.py \
    --data ${OUTPUT_DIR}/dataset_corrected \
    --model ${BASE_MODEL} \
    --batch 48 \
    --epochs 250 \
    --patience 150 \
    --imgsz 768 \
    --device 0 \
    --workers 4 \
    --hyp hyp_fish_small_dataset.yaml \
    --project ${OUTPUT_DIR}/runs \
    --name training \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "========================================================================"
echo "✓ EXPERIMENT 2 COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - Preprocessed dataset: ${OUTPUT_DIR}/dataset_corrected"
echo "  - Training results: ${OUTPUT_DIR}/runs/training"
echo "  - Best model: ${OUTPUT_DIR}/runs/training/weights/best.pt"
echo ""
echo "Key advantages over Experiment 1:"
echo "  - 67% fewer parameter updates (48 vs 16 batch)"
echo "  - Longer training with higher patience"
echo "  - Should have even more stable validation loss"
echo ""
echo "Compare with Experiment 1 to see if larger batch helps!"
echo "========================================================================"
