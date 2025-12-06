#!/usr/bin/env python3
"""
Compare Multiple YOLOv11 Models (Including Multi-Checkpoint Support)
Validates multiple models and shows comparison table

Supports:
- best.pt (best mAP@50)
- best_prec.pt (best precision)
- best_rec.pt (best recall)
- last.pt (last epoch)

Usage:
    # Compare specific experiments (all checkpoints)
    python compare_models_v2.py \
        --experiments runs/detect/medium_precision_v1_scratch runs/detect/medium_recall_v1_scratch \
        --data /path/to/data.yaml

    # Compare specific checkpoint types
    python compare_models_v2.py \
        --models runs/detect/*/weights/best_prec.pt \
        --data /path/to/data.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import glob
from collections import defaultdict


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
        
        # Get model size
        size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Get experiment name and checkpoint type
        checkpoint_name = model_path.stem  # e.g., 'best_prec', 'best', 'last'
        exp_name = model_path.parent.parent.name  # experiment directory name
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mAP50': mAP50,
            'mAP50_95': mAP50_95,
            'avg_acc': avg_acc,
            'size_mb': size_mb,
            'path': str(model_path),
            'checkpoint_type': checkpoint_name,
            'experiment': exp_name
        }
    except Exception as e:
        print(f"⚠ Error validating {model_path}: {e}")
        return None


def print_experiment_comparison(exp_metrics):
    """Print comparison for a single experiment across all checkpoint types"""
    exp_name = list(exp_metrics.keys())[0]
    checkpoints = exp_metrics[exp_name]
    
    if not checkpoints:
        return
    
    print(f"\n📊 Experiment: {exp_name}")
    print("─" * 90)
    print(f"{'Checkpoint':<15} {'Precision':>9} {'Recall':>9} {'F1':>9} {'mAP50':>9} {'AvgAcc':>9} {'Size':>8}")
    print("─" * 90)
    
    # Sort by checkpoint type priority
    priority = {'best': 0, 'best_prec': 1, 'best_rec': 2, 'last': 3}
    sorted_checkpoints = sorted(checkpoints, key=lambda x: priority.get(x['checkpoint_type'], 99))
    
    best_avg_acc = max(c['avg_acc'] for c in checkpoints)
    
    for m in sorted_checkpoints:
        ckpt_type = m['checkpoint_type']
        marker = "⭐" if m['avg_acc'] == best_avg_acc else "  "
        
        print(f"{marker} {ckpt_type:<13} {m['precision']*100:>8.2f}% {m['recall']*100:>8.2f}% "
              f"{m['f1']*100:>8.2f}% {m['mAP50']*100:>8.2f}% {m['avg_acc']*100:>8.2f}% {m['size_mb']:>7.1f}MB")
    
    # Recommendation
    best_ckpt = max(checkpoints, key=lambda x: x['avg_acc'])
    print(f"\n💡 Best checkpoint: {best_ckpt['checkpoint_type']} ({best_ckpt['avg_acc']*100:.2f}% avg acc)")


def print_overall_comparison(all_metrics, group_by_experiment=True):
    """Print overall comparison table"""
    if not all_metrics:
        print("❌ No valid metrics found!")
        return
    
    if group_by_experiment:
        # Group by experiment
        experiments = defaultdict(list)
        for m in all_metrics:
            experiments[m['experiment']].append(m)
        
        print("\n" + "=" * 90)
        print("📊 DETAILED COMPARISON BY EXPERIMENT")
        print("=" * 90)
        
        # Print each experiment's checkpoints
        for exp_name in sorted(experiments.keys()):
            exp_checkpoints = experiments[exp_name]
            print_experiment_comparison({exp_name: exp_checkpoints})
        
        # Print summary table with best checkpoint per experiment
        print("\n" + "=" * 90)
        print("📊 SUMMARY - BEST CHECKPOINT PER EXPERIMENT")
        print("=" * 90)
        print(f"{'Experiment':<35} {'Checkpoint':<12} {'Prec':>7} {'Rec':>7} {'AvgAcc':>7} {'Size':>8} {'Target':>6}")
        print("─" * 90)
        
        summary_data = []
        for exp_name in sorted(experiments.keys()):
            exp_checkpoints = experiments[exp_name]
            best_ckpt = max(exp_checkpoints, key=lambda x: x['avg_acc'])
            summary_data.append(best_ckpt)
        
        # Sort by avg_acc descending
        summary_data.sort(key=lambda x: x['avg_acc'], reverse=True)
        
        for m in summary_data:
            exp_name = m['experiment'][:33]
            ckpt_type = m['checkpoint_type'][:10]
            target_status = "✅" if m['avg_acc'] >= 0.70 else "❌"
            
            print(f"{exp_name:<35} {ckpt_type:<12} {m['precision']*100:>6.2f}% {m['recall']*100:>6.2f}% "
                  f"{m['avg_acc']*100:>6.2f}% {m['size_mb']:>7.1f}MB {target_status:>6}")
        
    else:
        # Flat comparison of all models
        all_metrics.sort(key=lambda x: x['avg_acc'], reverse=True)
        
        print("\n" + "=" * 100)
        print("📊 ALL MODELS COMPARISON (Sorted by Average Accuracy)")
        print("=" * 100)
        print(f"{'Experiment':<30} {'Ckpt':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'mAP50':>7} {'AvgAcc':>7} {'Size':>8} {'Target':>6}")
        print("─" * 100)
        
        for m in all_metrics:
            exp_name = m['experiment'][:28]
            ckpt_type = m['checkpoint_type'][:10]
            target_status = "✅" if m['avg_acc'] >= 0.70 else "❌"
            
            print(f"{exp_name:<30} {ckpt_type:<12} {m['precision']*100:>6.2f}% {m['recall']*100:>6.2f}% "
                  f"{m['f1']*100:>6.2f}% {m['mAP50']*100:>6.2f}% {m['avg_acc']*100:>6.2f}% "
                  f"{m['size_mb']:>7.1f}MB {target_status:>6}")
    
    # Overall best
    best_overall = max(all_metrics, key=lambda x: x['avg_acc'])
    
    print("\n" + "=" * 90)
    print("🏆 OVERALL BEST MODEL")
    print("=" * 90)
    print(f"Experiment:  {best_overall['experiment']}")
    print(f"Checkpoint:  {best_overall['checkpoint_type']}")
    print(f"Avg Acc:     {best_overall['avg_acc']*100:.2f}%")
    print(f"Precision:   {best_overall['precision']*100:.2f}%")
    print(f"Recall:      {best_overall['recall']*100:.2f}%")
    print(f"mAP@50:      {best_overall['mAP50']*100:.2f}%")
    print(f"Model Size:  {best_overall['size_mb']:.1f} MB")
    print(f"Path:        {best_overall['path']}")
    
    # Target analysis
    if best_overall['avg_acc'] >= 0.70:
        print(f"\n✅ TARGET ACHIEVED! ({best_overall['avg_acc']*100:.2f}% ≥ 70%)")
        if best_overall['size_mb'] <= 70:
            print(f"✅ SIZE CONSTRAINT MET! ({best_overall['size_mb']:.1f} MB ≤ 70 MB)")
            print(f"\n🎉 Ready for deployment!")
        else:
            print(f"⚠️  Size exceeds 70MB - consider quantization")
    else:
        gap = (0.70 - best_overall['avg_acc']) * 100
        print(f"\n📊 Gap to target: {gap:.2f}%")
        print(f"💡 Next steps:")
        print(f"   - Try larger models (medium/large)")
        print(f"   - Ensemble top 2-3 models")
        print(f"   - Extended training with lower LR")
    
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLOv11 models with multi-checkpoint support"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiments', type=str, nargs='+',
                       help='Experiment directories to compare (compares all checkpoints)')
    group.add_argument('--models', type=str, nargs='+',
                       help='Specific model .pt files (supports wildcards)')
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data.yaml file')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split (default: test)')
    parser.add_argument('--imgsz', type=int, default=768,
                       help='Input image size (default: 768)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--no-group', action='store_true',
                       help='Disable grouping by experiment')
    
    args = parser.parse_args()
    
    # Check data.yaml exists
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        return 1
    
    # Collect model paths
    model_paths = []
    
    if args.experiments:
        # Load all checkpoints from specified experiments
        checkpoint_types = ['best.pt', 'best_prec.pt', 'best_rec.pt', 'last.pt']
        
        for exp_dir in args.experiments:
            exp_path = Path(exp_dir)
            if not exp_path.exists():
                print(f"⚠ Warning: Experiment directory not found: {exp_dir}")
                continue
            
            weights_dir = exp_path / 'weights'
            if not weights_dir.exists():
                print(f"⚠ Warning: No weights directory in {exp_dir}")
                continue
            
            for ckpt_type in checkpoint_types:
                ckpt_path = weights_dir / ckpt_type
                if ckpt_path.exists():
                    model_paths.append(ckpt_path)
    
    else:  # args.models
        # Load specific model files
        for pattern in args.models:
            matches = glob.glob(pattern)
            if matches:
                model_paths.extend([Path(p) for p in matches if Path(p).exists()])
            else:
                p = Path(pattern)
                if p.exists():
                    model_paths.append(p)
    
    if len(model_paths) == 0:
        print(f"❌ Error: No models found")
        return 1
    
    # Remove duplicates
    model_paths = list(set(model_paths))
    
    # Print info
    print("=" * 90)
    print("MODEL COMPARISON WITH MULTI-CHECKPOINT SUPPORT")
    print("=" * 90)
    print(f"Models to compare: {len(model_paths)}")
    print(f"Dataset: {data_yaml}")
    print(f"Split:   {args.split}")
    print("=" * 90)
    
    # Validate each model
    all_metrics = []
    for i, model_path in enumerate(model_paths, 1):
        exp_name = model_path.parent.parent.name
        ckpt_type = model_path.stem
        print(f"[{i}/{len(model_paths)}] Validating: {exp_name}/{ckpt_type}")
        
        metrics = validate_single_model(
            model_path=model_path,
            data_yaml=data_yaml,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch
        )
        
        if metrics:
            all_metrics.append(metrics)
            print(f"    ✓ Avg Accuracy: {metrics['avg_acc']*100:.2f}%  "
                  f"Prec: {metrics['precision']*100:.2f}%  Rec: {metrics['recall']*100:.2f}%")
        else:
            print(f"    ✗ Failed")
    
    # Print comparison
    group_by_exp = not args.no_group and args.experiments is not None
    print_overall_comparison(all_metrics, group_by_experiment=group_by_exp)
    
    return 0


if __name__ == '__main__':
    exit(main())
