#!/bin/bash
# Comprehensive Testing Script for Best Model Checkpoint
# Tests the model on validation and test sets, generates metrics
#
# Usage:
#   bash test_best_model.sh <model_path> [data_yaml] [output_dir]
#
# Examples:
#   bash test_best_model.sh runs/detect/fish_s_moderate_s_v1/weights/best.pt
#   bash test_best_model.sh runs/detect/fish_s_moderate_s_v1/weights/best.pt dataset_root/data.yaml
#   bash test_best_model.sh runs/detect/fish_s_moderate_s_v1/weights/best.pt dataset_root/data.yaml my_test_results

# Parse command line arguments
if [ $# -lt 1 ]; then
    echo "❌ Error: Missing required argument"
    echo ""
    echo "Usage: bash test_best_model.sh <model_path> [data_yaml] [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  model_path  : Path to best.pt file (required)"
    echo "  data_yaml   : Path to data.yaml (default: dataset_root/data.yaml)"
    echo "  output_dir  : Output directory (default: test_results_<timestamp>)"
    echo ""
    echo "Examples:"
    echo "  bash test_best_model.sh runs/detect/fish_s_moderate_s_v1/weights/best.pt"
    echo "  bash test_best_model.sh runs/detect/fish_s_moderate_s_v1/weights/best.pt dataset_root/data.yaml"
    exit 1
fi

MODEL_PATH="$1"
DATA_YAML="${2:-dataset_root/data.yaml}"
OUTPUT_DIR="${3:-test_results_$(date +%Y%m%d_%H%M%S)}"

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

# Check if data.yaml exists
if [ ! -f "${DATA_YAML}" ]; then
    echo "❌ Error: data.yaml not found at ${DATA_YAML}"
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

# Extract and print metrics from results
echo "📊 FINAL METRICS SUMMARY"
echo "=========================================="

# Create Python script to extract metrics
python3 << 'PYTHON_SCRIPT'
import sys
import csv
from pathlib import Path

def extract_metrics(results_dir, split_name):
    """Extract metrics from results.csv"""
    results_csv = Path(results_dir) / 'results.csv'
    
    if not results_csv.exists():
        print(f"⚠ Warning: {results_csv} not found")
        return None
    
    try:
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            # Get last row (final metrics)
            rows = list(reader)
            if not rows:
                return None
            
            last_row = rows[-1]
            
            # Extract metrics (handle different column name formats)
            metrics = {}
            for key in last_row.keys():
                key_clean = key.strip()
                if 'precision' in key_clean.lower() and 'B' in key_clean:
                    metrics['precision'] = float(last_row[key])
                elif 'recall' in key_clean.lower() and 'B' in key_clean:
                    metrics['recall'] = float(last_row[key])
                elif 'mAP50-95' in key_clean and 'B' in key_clean:
                    metrics['mAP50-95'] = float(last_row[key])
                elif 'mAP50' in key_clean and 'B' in key_clean and '95' not in key_clean:
                    metrics['mAP50'] = float(last_row[key])
            
            return metrics
    except Exception as e:
        print(f"⚠ Error reading metrics: {e}")
        return None

def print_metrics(metrics, split_name):
    """Print metrics in a formatted way"""
    if not metrics:
        print(f"\n❌ {split_name.upper()} SET: No metrics found")
        return
    
    print(f"\n{split_name.upper()} SET:")
    print("-" * 50)
    
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    mAP50 = metrics.get('mAP50', 0)
    mAP50_95 = metrics.get('mAP50-95', 0)
    
    # Calculate F1 and average accuracy
    if precision > 0 and recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        avg_accuracy = (precision + recall) / 2
    else:
        f1 = 0
        avg_accuracy = 0
    
    print(f"  Precision:       {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:          {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:        {f1:.4f} ({f1*100:.2f}%)")
    print(f"  mAP@50:          {mAP50:.4f} ({mAP50*100:.2f}%)")
    print(f"  mAP@50-95:       {mAP50_95:.4f} ({mAP50_95*100:.2f}%)")
    print(f"  Avg Accuracy:    {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print("-" * 50)
    
    # Check if target met
    target = 0.70
    if avg_accuracy >= target:
        print(f"  ✅ TARGET MET! Average accuracy >= 70%")
    else:
        gap = (target - avg_accuracy) * 100
        print(f"  ❌ Gap to 70% target: {gap:.1f}%")

# Get output directory from command line
import sys
if len(sys.argv) > 1:
    output_dir = sys.argv[1]
else:
    output_dir = 'test_results_best_model'

# Extract and print validation metrics
val_metrics = extract_metrics(f"{output_dir}/validation_results", "validation")
print_metrics(val_metrics, "VALIDATION")

# Extract and print test metrics
test_metrics = extract_metrics(f"{output_dir}/test_results", "test")
print_metrics(test_metrics, "TEST")

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Detailed results available at:"
echo "  Validation: ${OUTPUT_DIR}/validation_results/"
echo "  Test:       ${OUTPUT_DIR}/test_results/"
echo "  Inference:  ${OUTPUT_DIR}/inference_samples/"
echo ""
echo "Key files to review:"
echo "  - confusion_matrix.png (per-class performance)"
echo "  - PR_curve.png (precision-recall curve)"
echo "  - F1_curve.png (F1 score vs confidence)"
echo "  - results.csv (detailed metrics)"
echo "=========================================="
