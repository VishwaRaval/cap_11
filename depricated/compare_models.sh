#!/bin/bash
# Compare Multiple Models Side-by-Side
# Validates multiple models and shows comparison table
#
# Usage:
#   bash compare_models.sh model1.pt model2.pt [model3.pt ...] [data_yaml] [split]
#
# Examples:
#   bash compare_models.sh runs/detect/*/weights/best.pt
#   bash compare_models.sh run1/best.pt run2/best.pt run3/best.pt dataset_root/data.yaml test

if [ $# -lt 2 ]; then
    echo "❌ Error: Need at least 2 models to compare"
    echo ""
    echo "Usage: bash compare_models.sh model1.pt model2.pt [model3.pt ...] [data_yaml] [split]"
    echo ""
    echo "Examples:"
    echo "  bash compare_models.sh runs/detect/*/weights/best.pt"
    echo "  bash compare_models.sh run1/best.pt run2/best.pt dataset_root/data.yaml test"
    exit 1
fi

# Parse arguments
MODELS=()
DATA_YAML="dataset_root/data.yaml"
SPLIT="test"

for arg in "$@"; do
    if [[ "$arg" == *.pt ]]; then
        if [ -f "$arg" ]; then
            MODELS+=("$arg")
        else
            echo "⚠ Warning: Model not found: $arg (skipping)"
        fi
    elif [[ "$arg" == *.yaml ]]; then
        DATA_YAML="$arg"
    elif [[ "$arg" =~ ^(train|val|test)$ ]]; then
        SPLIT="$arg"
    fi
done

if [ ${#MODELS[@]} -lt 2 ]; then
    echo "❌ Error: Need at least 2 valid model files"
    exit 1
fi

echo "=========================================="
echo "MODEL COMPARISON"
echo "=========================================="
echo "Models to compare: ${#MODELS[@]}"
for i in "${!MODELS[@]}"; do
    echo "  $((i+1)). ${MODELS[$i]}"
done
echo "Dataset: ${DATA_YAML}"
echo "Split:   ${SPLIT}"
echo "=========================================="
echo ""

# Validate each model and collect results
RESULTS_DIR="comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "🔄 Validating models..."
echo ""

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NAME="model_$((i+1))"
    
    echo "[$((i+1))/${#MODELS[@]}] Validating: $(basename $MODEL)"
    
    yolo val \
        model="$MODEL" \
        data="$DATA_YAML" \
        split="$SPLIT" \
        imgsz=768 \
        batch=16 \
        save_json=False \
        plots=False \
        project="$RESULTS_DIR" \
        name="$MODEL_NAME" \
        verbose=False > /dev/null 2>&1
    
    echo "    ✓ Complete"
done

echo ""
echo "=========================================="
echo "📊 COMPARISON RESULTS"
echo "=========================================="

# Extract and compare metrics
python3 << 'PYTHON_SCRIPT'
import csv
import sys
from pathlib import Path

def extract_metrics(results_dir):
    """Extract metrics from results.csv"""
    results_csv = Path(results_dir) / 'results.csv'
    
    if not results_csv.exists():
        return None
    
    try:
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return None
            
            last_row = rows[-1]
            
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
            
            if metrics.get('precision') and metrics.get('recall'):
                p, r = metrics['precision'], metrics['recall']
                metrics['f1'] = 2 * (p * r) / (p + r)
                metrics['avg_acc'] = (p + r) / 2
            
            return metrics
    except:
        return None

# Get results directory and model paths from environment
results_dir = sys.argv[1] if len(sys.argv) > 1 else "comparison_results"
model_paths = sys.argv[2:] if len(sys.argv) > 2 else []

# Collect all metrics
all_metrics = []
for i, model_path in enumerate(model_paths):
    model_name = f"model_{i+1}"
    metrics = extract_metrics(f"{results_dir}/{model_name}")
    if metrics:
        all_metrics.append({
            'name': Path(model_path).parent.parent.name,  # Get experiment name
            'path': model_path,
            **metrics
        })

if not all_metrics:
    print("❌ No metrics found!")
    sys.exit(1)

# Print comparison table
print("")
print("┌─" + "─" * 78 + "─┐")
print(f"│ {'MODEL':<25} {'PREC':>8} {'REC':>8} {'F1':>8} {'mAP50':>8} {'AVG_ACC':>8} {'TARGET':>7} │")
print("├─" + "─" * 78 + "─┤")

best_avg_acc = 0
best_model_idx = 0

for i, m in enumerate(all_metrics):
    name = m['name'][:23]  # Truncate long names
    prec = m.get('precision', 0)
    rec = m.get('recall', 0)
    f1 = m.get('f1', 0)
    map50 = m.get('mAP50', 0)
    avg_acc = m.get('avg_acc', 0)
    
    # Track best model
    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        best_model_idx = i
    
    target_status = "✅" if avg_acc >= 0.70 else "❌"
    
    print(f"│ {name:<25} {prec*100:>7.2f}% {rec*100:>7.2f}% {f1*100:>7.2f}% {map50*100:>7.2f}% {avg_acc*100:>7.2f}% {target_status:>7} │")

print("└─" + "─" * 78 + "─┘")
print("")

# Print best model
best = all_metrics[best_model_idx]
print(f"🏆 BEST MODEL: {best['name']}")
print(f"   Average Accuracy: {best['avg_acc']*100:.2f}%")
print(f"   Recall: {best['recall']*100:.2f}%  Precision: {best['precision']*100:.2f}%")
print(f"   Model: {best['path']}")
print("")

# Show improvement recommendations
if best['avg_acc'] >= 0.70:
    print("✅ Target achieved! Ready for deployment.")
else:
    gap = (0.70 - best['avg_acc']) * 100
    print(f"📊 Best model is {gap:.1f}% away from 70% target")
    print("")
    print("💡 Next steps:")
    if gap <= 2:
        print("   - Try dehazing/CLAHE preprocessing")
        print("   - Adjust inference confidence threshold")
    elif gap <= 5:
        print("   - Add dehazing preprocessing")
        print("   - Try YOLOv11m for more capacity")
    else:
        print("   - Use YOLOv11m (larger model)")
        print("   - Add preprocessing (dehaze + CLAHE)")

PYTHON_SCRIPT

# Pass arguments to Python script
python3 -c "import sys; print(''); exec(open('/dev/stdin').read())" "$RESULTS_DIR" "${MODELS[@]}" < <(cat << 'PYTHON_SCRIPT'
import csv
import sys
from pathlib import Path

def extract_metrics(results_dir):
    """Extract metrics from results.csv"""
    results_csv = Path(results_dir) / 'results.csv'
    
    if not results_csv.exists():
        return None
    
    try:
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return None
            
            last_row = rows[-1]
            
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
            
            if metrics.get('precision') and metrics.get('recall'):
                p, r = metrics['precision'], metrics['recall']
                metrics['f1'] = 2 * (p * r) / (p + r)
                metrics['avg_acc'] = (p + r) / 2
            
            return metrics
    except:
        return None

# Get arguments
results_dir = sys.argv[1]
model_paths = sys.argv[2:]

# Collect metrics
all_metrics = []
for i, model_path in enumerate(model_paths):
    model_name = f"model_{i+1}"
    metrics = extract_metrics(f"{results_dir}/{model_name}")
    if metrics:
        all_metrics.append({
            'name': Path(model_path).parent.parent.name,
            'path': model_path,
            **metrics
        })

if not all_metrics:
    print("❌ No metrics found!")
    sys.exit(1)

# Print table
print("")
print("┌─" + "─" * 78 + "─┐")
print(f"│ {'MODEL':<25} {'PREC':>8} {'REC':>8} {'F1':>8} {'mAP50':>8} {'AVG_ACC':>8} {'TARGET':>7} │")
print("├─" + "─" * 78 + "─┤")

best_avg_acc = 0
best_model_idx = 0

for i, m in enumerate(all_metrics):
    name = m['name'][:23]
    prec = m.get('precision', 0)
    rec = m.get('recall', 0)
    f1 = m.get('f1', 0)
    map50 = m.get('mAP50', 0)
    avg_acc = m.get('avg_acc', 0)
    
    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        best_model_idx = i
    
    target_status = "✅" if avg_acc >= 0.70 else "❌"
    print(f"│ {name:<25} {prec*100:>7.2f}% {rec*100:>7.2f}% {f1*100:>7.2f}% {map50*100:>7.2f}% {avg_acc*100:>7.2f}% {target_status:>7} │")

print("└─" + "─" * 78 + "─┘")
print("")

# Best model
best = all_metrics[best_model_idx]
print(f"🏆 BEST MODEL: {best['name']}")
print(f"   Average Accuracy: {best['avg_acc']*100:.2f}%")
print(f"   Recall: {best['recall']*100:.2f}%  Precision: {best['precision']*100:.2f}%")
print(f"   Model: {best['path']}")
print("")

# Recommendations
if best['avg_acc'] >= 0.70:
    print("✅ Target achieved! Ready for deployment.")
else:
    gap = (0.70 - best['avg_acc']) * 100
    print(f"📊 Best model is {gap:.1f}% away from 70% target")
    print("")
    print("💡 Next steps:")
    if gap <= 2:
        print("   - Try dehazing/CLAHE preprocessing")
        print("   - Adjust inference confidence threshold")
    elif gap <= 5:
        print("   - Add dehazing preprocessing")
        print("   - Try YOLOv11m for more capacity")
    else:
        print("   - Use YOLOv11m (larger model)")
        print("   - Add preprocessing (dehaze + CLAHE)")

PYTHON_SCRIPT
)

echo ""
echo "=========================================="
echo "Detailed results saved to: $RESULTS_DIR"
echo "=========================================="
