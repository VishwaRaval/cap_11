#!/bin/bash
# Comprehensive Testing Script for Best Model Checkpoint
# Tests the model on validation and test sets, generates metrics

MODEL_PATH="runs/detect/fish_s_recall_optimized_s_v1/weights/best.pt"
DATA_YAML="dataset_root/data.yaml"
OUTPUT_DIR="test_results_best_model"

echo "=========================================="
echo "COMPREHENSIVE MODEL TESTING"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATA_YAML}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "❌ Error: Model not found at ${MODEL_PATH}"
    echo "Please check the path and try again"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Test 1: Validation Set Performance
echo "📊 Test 1: Validation Set Evaluation"
echo "--------------------------------------"
yolo val \
    model="${MODEL_PATH}" \
    data="${DATA_YAML}" \
    split=val \
    imgsz=768 \
    batch=16 \
    save_json=True \
    plots=True \
    project="${OUTPUT_DIR}" \
    name=validation_results

echo ""
echo "✓ Validation complete!"
echo ""

# Test 2: Test Set Performance
echo "📊 Test 2: Test Set Evaluation"
echo "--------------------------------------"
yolo val \
    model="${MODEL_PATH}" \
    data="${DATA_YAML}" \
    split=test \
    imgsz=768 \
    batch=16 \
    save_json=True \
    plots=True \
    project="${OUTPUT_DIR}" \
    name=test_results

echo ""
echo "✓ Test set evaluation complete!"
echo ""

# Test 3: Sample Inference on Test Images
echo "📊 Test 3: Visual Inference on Sample Images"
echo "--------------------------------------"
python infer_edge.py \
    --model "${MODEL_PATH}" \
    --source dataset_root/test/images \
    --output "${OUTPUT_DIR}/inference_samples" \
    --conf 0.25 \
    --iou 0.45 \
    --save-json \
    --no-display

echo ""
echo "✓ Inference complete!"
echo ""

# Summary
echo "=========================================="
echo "TESTING COMPLETE"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Check these files:"
echo "  1. Validation metrics: ${OUTPUT_DIR}/validation_results/"
echo "  2. Test set metrics: ${OUTPUT_DIR}/test_results/"
echo "  3. Inference samples: ${OUTPUT_DIR}/inference_samples/"
echo ""
echo "Key files to review:"
echo "  - confusion_matrix.png (per-class performance)"
echo "  - PR_curve.png (precision-recall curve)"
echo "  - F1_curve.png (F1 score vs confidence)"
echo "  - results.csv (detailed metrics)"
echo "=========================================="
