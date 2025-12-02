#!/usr/bin/env python3
"""
Compare Multiple YOLOv11 Models
Validates multiple models and shows comparison table

Usage:
    python compare_models.py \
        --models model1.pt model2.pt model3.pt \
        --data /path/to/data.yaml \
        --split test
        
Example:
    python compare_models.py \
        --models runs/detect/*/weights/best.pt \
        --data /scratch/am14419/projects/cap_11/dataset_root/data.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import glob


def validate_single_model(model_path, data_yaml, split, imgsz=768, batch=16):
    """Validate a single model and return metrics"""
    try:
        model = YOLO(str(model_path))
        results = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            batch=batch,
            plots=False,
            save_json=False,
            verbose=False
        )
        
        metrics = results.results_dict
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        mAP50 = metrics.get('metrics/mAP50(B)', 0)
        mAP50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        
        if precision > 0 and recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            avg_acc = (precision + recall) / 2
        else:
            f1 = 0
            avg_acc = 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mAP50': mAP50,
            'mAP50_95': mAP50_95,
            'avg_acc': avg_acc,
            'path': str(model_path),
            'name': model_path.parent.parent.name  # Get experiment name
        }
    except Exception as e:
        print(f"⚠ Error validating {model_path}: {e}")
        return None


def print_comparison_table(all_metrics):
    """Print comparison table"""
    if not all_metrics:
        print("❌ No valid metrics found!")
        return
    
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON")
    print("="*80)
    print("")
    print("┌─" + "─" * 78 + "─┐")
    print(f"│ {'MODEL':<25} {'PREC':>8} {'REC':>8} {'F1':>8} {'mAP50':>8} {'AVG_ACC':>8} {'TARGET':>7} │")
    print("├─" + "─" * 78 + "─┤")
    
    best_avg_acc = 0
    best_model_idx = 0
    
    for i, m in enumerate(all_metrics):
        name = m['name'][:23]  # Truncate long names
        prec = m['precision']
        rec = m['recall']
        f1 = m['f1']
        map50 = m['mAP50']
        avg_acc = m['avg_acc']
        
        # Track best
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
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple YOLOv11 models"
    )
    
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Paths to model .pt files (supports wildcards)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data.yaml file')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to validate on (default: test)')
    parser.add_argument('--imgsz', type=int, default=768,
                       help='Input image size (default: 768)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    
    args = parser.parse_args()
    
    # Check data.yaml exists
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        return 1
    
    # Expand wildcards and collect model paths
    model_paths = []
    for pattern in args.models:
        matches = glob.glob(pattern)
        if matches:
            model_paths.extend([Path(p) for p in matches if Path(p).exists()])
        else:
            # Try as direct path
            p = Path(pattern)
            if p.exists():
                model_paths.append(p)
    
    if len(model_paths) < 2:
        print(f"❌ Error: Need at least 2 models to compare")
        print(f"   Found: {len(model_paths)} model(s)")
        return 1
    
    # Print info
    print("="*80)
    print("MODEL COMPARISON STARTING")
    print("="*80)
    print(f"Models to compare: {len(model_paths)}")
    for i, p in enumerate(model_paths, 1):
        print(f"  {i}. {p}")
    print(f"Dataset: {data_yaml}")
    print(f"Split:   {args.split}")
    print("="*80)
    print("")
    
    # Validate each model
    all_metrics = []
    for i, model_path in enumerate(model_paths, 1):
        print(f"[{i}/{len(model_paths)}] Validating: {model_path.name}")
        
        metrics = validate_single_model(
            model_path=model_path,
            data_yaml=data_yaml,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch
        )
        
        if metrics:
            all_metrics.append(metrics)
            print(f"    ✓ Avg Accuracy: {metrics['avg_acc']*100:.2f}%")
        else:
            print(f"    ✗ Failed")
        print("")
    
    # Print comparison
    print_comparison_table(all_metrics)
    
    return 0


if __name__ == '__main__':
    exit(main())
