#!/bin/bash
# Quick Validation Script - Just prints metrics for a model
# Much faster than full testing (no inference on all images)
#
# Usage:
#   bash quick_validate.sh <model_path> [data_yaml] [split]
#
# Examples:
#   bash quick_validate.sh runs/detect/fish_s_moderate_s_v1/weights/best.pt
#   bash quick_validate.sh runs/detect/fish_s_moderate_s_v1/weights/best.pt dataset_root/data.yaml test

# Parse command line arguments
if [ $# -lt 1 ]; then
    echo "❌ Error: Missing required argument"
    echo ""
    echo "Usage: bash quick_validate.sh <model_path> [data_yaml] [split]"
    echo ""
    echo "Arguments:"
    echo "  model_path  : Path to .pt file (required)"
    echo "  data_yaml   : Path to data.yaml (default: dataset_root/data.yaml)"
    echo "  split       : Dataset split to validate (default: test, options: train/val/test)"
    echo ""
    echo "Examples:"
    echo "  bash quick_validate.sh runs/detect/fish_s_moderate_s_v1/weights/best.pt"
    echo "  bash quick_validate.sh runs/detect/fish_s_moderate_s_v1/weights/best.pt dataset_root/data.yaml val"
    exit 1
fi

MODEL_PATH="$1"
DATA_YAML="${2:-dataset_root/data.yaml}"
SPLIT="${3:-test}"
OUTPUT_DIR="quick_val_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "QUICK MODEL VALIDATION"
echo "=========================================="
echo "Model:    ${MODEL_PATH}"
echo "Dataset:  ${DATA_YAML}"
echo "Split:    ${SPLIT}"
echo "=========================================="
echo ""

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "❌ Error: Model not found at ${MODEL_PATH}"
    exit 1
fi

# Check if data.yaml exists
if [ ! -f "${DATA_YAML}" ]; then
    echo "❌ Error: data.yaml not found at ${DATA_YAML}"
    exit 1
fi

# Run validation
echo "🔄 Running validation on ${SPLIT} set..."
echo ""

yolo val \
    model="${MODEL_PATH}" \
    data="${DATA_YAML}" \
    split="${SPLIT}" \
    imgsz=768 \
    batch=16 \
    save_json=False \
    plots=True \
    project="${OUTPUT_DIR}" \
    name="${SPLIT}_results" \
    verbose=False

echo ""
echo "=========================================="
echo "📊 METRICS SUMMARY"
echo "=========================================="

# Extract and print metrics
python3 << PYTHON_SCRIPT
import csv
from pathlib import Path

def extract_and_print_metrics(results_dir):
    """Extract and print metrics from results.csv"""
    results_csv = Path(results_dir) / 'results.csv'
    
    if not results_csv.exists():
        print(f"⚠ Warning: {results_csv} not found")
        return
    
    try:
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                print("❌ No metrics found in results.csv")
                return
            
            last_row = rows[-1]
            
            # Extract metrics
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
            
            if not metrics:
                print("❌ Could not extract metrics from results.csv")
                return
            
            # Calculate derived metrics
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            mAP50 = metrics.get('mAP50', 0)
            mAP50_95 = metrics.get('mAP50-95', 0)
            
            if precision > 0 and recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                avg_accuracy = (precision + recall) / 2
            else:
                f1 = 0
                avg_accuracy = 0
            
            # Print formatted metrics
            print("")
            print("┌" + "─" * 48 + "┐")
            print(f"│ {'METRIC':<20} {'VALUE':>12} {'PERCENTAGE':>13} │")
            print("├" + "─" * 48 + "┤")
            print(f"│ {'Precision':<20} {precision:>12.4f} {precision*100:>12.2f}% │")
            print(f"│ {'Recall':<20} {recall:>12.4f} {recall*100:>12.2f}% │")
            print(f"│ {'F1 Score':<20} {f1:>12.4f} {f1*100:>12.2f}% │")
            print("├" + "─" * 48 + "┤")
            print(f"│ {'mAP@50':<20} {mAP50:>12.4f} {mAP50*100:>12.2f}% │")
            print(f"│ {'mAP@50-95':<20} {mAP50_95:>12.4f} {mAP50_95*100:>12.2f}% │")
            print("├" + "─" * 48 + "┤")
            print(f"│ {'Avg Accuracy':<20} {avg_accuracy:>12.4f} {avg_accuracy*100:>12.2f}% │")
            print("└" + "─" * 48 + "┘")
            print("")
            
            # Check target
            target = 0.70
            if avg_accuracy >= target:
                print("✅ TARGET MET! Average accuracy >= 70%")
            else:
                gap = (target - avg_accuracy) * 100
                print(f"📊 Gap to 70% target: {gap:.1f}%")
                
                # Provide recommendation
                if gap <= 2:
                    print("💡 Very close! Try:")
                    print("   1. Dehazing + CLAHE preprocessing")
                    print("   2. Slightly lower confidence threshold at inference")
                elif gap <= 5:
                    print("💡 Recommendations:")
                    print("   1. Try dehazing preprocessing")
                    print("   2. Or switch to YOLOv11m for more capacity")
                else:
                    print("💡 Recommendations:")
                    print("   1. Try YOLOv11m (larger model)")
                    print("   2. Add dehazing + CLAHE preprocessing")
                    print("   3. Check for data quality issues")
            
    except Exception as e:
        print(f"❌ Error extracting metrics: {e}")
        import traceback
        traceback.print_exc()

# Run extraction
extract_and_print_metrics("$OUTPUT_DIR/${SPLIT}_results")

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Detailed results saved to:"
echo "  ${OUTPUT_DIR}/${SPLIT}_results/"
echo ""
echo "View these files for more details:"
echo "  - confusion_matrix.png"
echo "  - PR_curve.png"
echo "  - F1_curve.png"
echo "=========================================="
