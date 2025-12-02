#!/usr/bin/env python3
"""
Simple Model Validation Script
Validates a YOLOv11 model and prints metrics in a nice format

Usage:
    python validate_model.py --model <path_to_best.pt> --data <path_to_data.yaml> [--split test]
    
Example:
    python validate_model.py \
        --model runs/detect/fish_s_moderate_s_v1/weights/best.pt \
        --data /scratch/am14419/projects/cap_11/dataset_root/data.yaml \
        --split test
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def print_metrics_table(results):
    """Print metrics in a formatted table"""
    # Extract metrics from results
    metrics = results.results_dict
    
    # Get box metrics (these are the ones we care about)
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    mAP50 = metrics.get('metrics/mAP50(B)', 0)
    mAP50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    
    # Calculate derived metrics
    if precision > 0 and recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        avg_accuracy = (precision + recall) / 2
    else:
        f1 = 0
        avg_accuracy = 0
    
    # Print formatted table
    print("\n" + "="*70)
    print("📊 VALIDATION METRICS")
    print("="*70)
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
    
    # Check against target
    target = 0.70
    if avg_accuracy >= target:
        print("✅ TARGET MET! Average accuracy >= 70%")
        print("   Ready for deployment!")
    else:
        gap = (target - avg_accuracy) * 100
        print(f"📊 Gap to 70% target: {gap:.1f}%")
        print("")
        
        # Provide recommendations
        if gap <= 2:
            print("💡 Very close! Try:")
            print("   1. Dehazing + CLAHE preprocessing")
            print("   2. Lower confidence threshold at inference (0.20 instead of 0.25)")
        elif gap <= 5:
            print("💡 Recommendations:")
            print("   1. Add dehazing preprocessing")
            print("   2. Try YOLOv11m for more capacity")
        else:
            print("💡 Recommendations:")
            print("   1. Use YOLOv11m (larger model)")
            print("   2. Add dehazing + CLAHE preprocessing")
            print("   3. Check dataset quality")
    
    print("="*70)
    
    return avg_accuracy


def validate_model(model_path, data_yaml, split='test', imgsz=768, batch=16):
    """Run validation on a model"""
    
    # Check files exist
    model_path = Path(model_path)
    data_yaml = Path(data_yaml)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    
    # Print info
    print("="*70)
    print("MODEL VALIDATION")
    print("="*70)
    print(f"Model:    {model_path}")
    print(f"Dataset:  {data_yaml}")
    print(f"Split:    {split}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print("="*70)
    print("")
    
    # Load model
    print("🔄 Loading model...")
    model = YOLO(str(model_path))
    
    # Get model size
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Model loaded: {model_size_mb:.2f} MB")
    print("")
    
    # Run validation
    print(f"🔄 Running validation on {split} set...")
    print("")
    
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        plots=True,
        save_json=False,
        verbose=True
    )
    
    # Print metrics
    avg_acc = print_metrics_table(results)
    
    return results, avg_acc


def main():
    parser = argparse.ArgumentParser(
        description="Validate YOLOv11 model and print metrics"
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model .pt file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data.yaml file')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to validate on (default: test)')
    parser.add_argument('--imgsz', type=int, default=768,
                       help='Input image size (default: 768)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    
    args = parser.parse_args()
    
    try:
        results, avg_acc = validate_model(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch
        )
        
        print(f"\n✓ Validation complete!")
        print(f"  Average Accuracy: {avg_acc*100:.2f}%")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
