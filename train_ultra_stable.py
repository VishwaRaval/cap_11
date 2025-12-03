#!/usr/bin/env python3
"""
ULTRA-STABLE YOLOv11 Training for Final Push to 70%

Strategy:
- Large batch size (64) for stable gradients
- Very low learning rate (0.0001) for minimal updates
- AdamW optimizer for better convergence
- Aggressive early stopping to prevent overfitting
- Minimal augmentation to preserve learned features

Usage:
    python train_ultra_stable.py \
        --data /scratch/am14419/projects/cap_11/dataset_root \
        --model /scratch/am14419/projects/cap_11/runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt \
        --batch 64 \
        --name ultra_stable_v1
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import os
from ultralytics import YOLO


def update_data_yaml(dataset_root):
    """Update data.yaml with absolute paths."""
    dataset_root = Path(dataset_root).resolve()
    data_yaml_path = dataset_root / 'data.yaml'
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_root}")
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Set absolute paths
    data_config['path'] = str(dataset_root)
    data_config['train'] = 'train/images'
    data_config['val'] = 'valid/images'
    data_config['test'] = 'test/images'
    
    if 'nc' not in data_config:
        data_config['nc'] = len(data_config.get('names', []))
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✓ Updated data.yaml with absolute paths")
    print(f"  Dataset root: {dataset_root}")
    print(f"  Classes: {data_config['nc']} - {data_config.get('names', [])}")
    
    return data_yaml_path


def train_ultra_stable(args):
    """
    Train YOLOv11 with ultra-stable settings for final accuracy push.
    """
    # Validate paths
    dataset_root = Path(args.data).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")
    
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    # Set up output directory with ABSOLUTE PATH
    output_base = Path("/scratch/am14419/projects/cap_11/runs/detect").resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Output directory: {output_base}")
    
    # Update data.yaml
    data_yaml = update_data_yaml(dataset_root)
    
    # Load model
    print(f"\n✓ Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    # Hyperparameters path
    hyp_path = Path(args.hyp).resolve() if args.hyp else Path("hyp_fish_ultra_stable.yaml").resolve()
    if not hyp_path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {hyp_path}")
    
    print(f"✓ Using hyperparameters: {hyp_path}")
    
    # Training configuration - ULTRA STABLE
    train_config = {
        # Data
        'data': str(data_yaml),
        
        # Training schedule
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': 768,
        
        # Hardware
        'device': args.device,
        'workers': args.workers,
        'cache': False,  # Don't cache (memory intensive)
        
        # Output paths (ABSOLUTE)
        'project': str(output_base),
        'name': args.name,
        'exist_ok': False,
        
        # Optimizer - AdamW for better convergence
        'optimizer': 'AdamW',  # CRITICAL: Better than SGD for small datasets
        
        # Learning rate schedule
        'cos_lr': True,  # Cosine scheduling (worked before!)
        'lr0': 0.0001,   # Override hyp file if needed
        'lrf': 0.00001,  # Final LR
        
        # Regularization
        'weight_decay': 0.0015,
        'dropout': 0.25,
        'label_smoothing': 0.15,
        
        # Early stopping - AGGRESSIVE
        'patience': args.patience,  # Stop early if no improvement
        
        # Augmentation control
        'close_mosaic': 0,  # Mosaic disabled in hyp file
        
        # Validation
        'val': True,
        'plots': True,
        'save': True,
        'save_period': -1,  # Only save best
        
        # Stability
        'deterministic': True,
        'seed': 42,
        'verbose': True,
        'amp': True,  # Automatic mixed precision
        
        # Model settings
        'pretrained': False,  # We're fine-tuning
        'single_cls': False,
        'rect': False,  # No rectangular training
        'resume': False,
        'freeze': None,
        
        # Hyperparameters file
        'cfg': str(hyp_path),
    }
    
    # Print configuration
    print("\n" + "=" * 80)
    print("ULTRA-STABLE TRAINING CONFIGURATION FOR 70% TARGET")
    print("=" * 80)
    print(f"\n🎯 GOAL: Stable training → 70%+ accuracy")
    print(f"\n📊 KEY STRATEGIES:")
    print(f"  ✓ Large batch ({args.batch}) → Stable gradients, fewer updates")
    print(f"  ✓ Very low LR (0.0001) → Minimal parameter changes")
    print(f"  ✓ AdamW optimizer → Better convergence than SGD")
    print(f"  ✓ Minimal augmentation → Preserve learned features")
    print(f"  ✓ Strong regularization → Prevent overfitting")
    print(f"  ✓ Aggressive early stop (patience={args.patience}) → Stop at first plateau")
    print(f"\n📁 PATHS:")
    print(f"  Dataset: {dataset_root}")
    print(f"  Base model: {model_path}")
    print(f"  Output: {output_base / args.name}")
    print(f"\n⚙️  TRAINING PARAMETERS:")
    for key in sorted(train_config.keys()):
        value = train_config[key]
        print(f"  {key:20s}: {value}")
    print("=" * 80 + "\n")
    
    # Start training
    print("🚀 Starting ultra-stable training...\n")
    print("📈 Expected behavior:")
    print("  - Smooth, low-variance training curves")
    print("  - Slower convergence (more epochs needed)")
    print("  - Better generalization (val vs train closer)")
    print("  - Will auto-stop if no improvement\n")
    
    results = model.train(**train_config)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    
    # Find results directory
    results_dir = output_base / args.name
    
    # Post-process metrics
    post_process_metrics(results_dir)
    
    print(f"\n📊 Results saved to: {results_dir}")
    print(f"📁 Best weights: {results_dir / 'weights' / 'best.pt'}")
    print(f"\n💡 Next steps:")
    print(f"  1. Check if accuracy >= 70%")
    print(f"  2. Check model size < 75MB")
    print(f"  3. Compare with baseline (64.64%)")
    
    return results_dir


def post_process_metrics(results_dir):
    """Create comprehensive metrics summary."""
    results_csv = results_dir / 'results.csv'
    if not results_csv.exists():
        print(f"⚠ Warning: results.csv not found in {results_dir}")
        return
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # Column mapping
    col_mapping = {
        'epoch': 'epoch',
        'train/box_loss': 'train_box_loss',
        'train/cls_loss': 'train_cls_loss',
        'train/dfl_loss': 'train_dfl_loss',
        'val/box_loss': 'val_box_loss',
        'val/cls_loss': 'val_cls_loss',
        'val/dfl_loss': 'val_dfl_loss',
        'metrics/precision(B)': 'precision',
        'metrics/recall(B)': 'recall',
        'metrics/mAP50(B)': 'mAP50',
        'metrics/mAP50-95(B)': 'mAP50_95',
    }
    
    summary_data = {}
    for orig_col, new_col in col_mapping.items():
        if orig_col in df.columns:
            summary_data[new_col] = df[orig_col]
    
    if not summary_data:
        print(f"⚠ Warning: No matching columns found in results.csv")
        return
    
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate derived metrics
    if 'precision' in summary_df.columns and 'recall' in summary_df.columns:
        # F1 score
        summary_df['f1_score'] = (
            2 * (summary_df['precision'] * summary_df['recall']) / 
            (summary_df['precision'] + summary_df['recall'] + 1e-8)
        )
        
        # Average accuracy (simple mean of precision and recall)
        summary_df['avg_accuracy'] = (summary_df['precision'] + summary_df['recall']) / 2
        
        # Gap to 70% target
        summary_df['gap_to_70pct'] = 0.70 - summary_df['avg_accuracy']
        
        # Training stability metrics
        if 'val_box_loss' in summary_df.columns:
            # Rolling std of validation loss (last 10 epochs)
            summary_df['val_loss_std_10ep'] = (
                summary_df['val_box_loss'].rolling(window=10, min_periods=1).std()
            )
    
    # Save summary
    summary_csv = results_dir / 'metrics_summary.csv'
    summary_df.to_csv(summary_csv, index=False, float_format='%.6f')
    
    print(f"\n✓ Created metrics summary: {summary_csv}")
    
    # Print final metrics
    if len(summary_df) > 0:
        final = summary_df.iloc[-1]
        best_idx = summary_df['avg_accuracy'].idxmax() if 'avg_accuracy' in summary_df else -1
        best = summary_df.iloc[best_idx] if best_idx >= 0 else final
        
        print("\n" + "=" * 80)
        print("📈 FINAL METRICS")
        print("=" * 80)
        print(f"\nFinal Epoch ({int(final['epoch'])}):")
        print("-" * 50)
        for col in ['precision', 'recall', 'f1_score', 'mAP50', 'mAP50_95', 'avg_accuracy']:
            if col in final:
                value = final[col]
                if pd.notna(value):
                    print(f"  {col:20s}: {value:.2%}")
        
        if 'avg_accuracy' in best and 'epoch' in best:
            print(f"\nBest Epoch ({int(best['epoch'])}):")
            print("-" * 50)
            for col in ['precision', 'recall', 'f1_score', 'mAP50', 'mAP50_95', 'avg_accuracy']:
                if col in best:
                    value = best[col]
                    if pd.notna(value):
                        print(f"  {col:20s}: {value:.2%}")
        
        print("=" * 80)
        
        # Check target achievement
        if 'avg_accuracy' in best:
            avg_acc = best['avg_accuracy']
            target_met = avg_acc >= 0.70
            
            print(f"\n🎯 TARGET ACHIEVEMENT:")
            print(f"  Current: {avg_acc:.2%}")
            print(f"  Target:  70.00%")
            print(f"  Status:  {'✅ TARGET MET!' if target_met else f'❌ Gap: {(0.70-avg_acc)*100:.2f}%'}")
            
            if not target_met:
                print(f"\n💡 SUGGESTIONS IF TARGET NOT MET:")
                print(f"  - Try even larger batch (e.g., 80 or 96)")
                print(f"  - Further reduce LR (e.g., 0.00005)")
                print(f"  - Train longer (current: {int(final['epoch'])} epochs)")
                print(f"  - Check if model is underfitting (val loss still decreasing)")


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-stable YOLOv11 training for final push to 70%"
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset root')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to .pt weights for fine-tuning')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Maximum training epochs (default: 300)')
    parser.add_argument('--batch', type=int, default=64,
                       help='Batch size (default: 64 for stability)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device (default: 0)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Dataloader workers (default: 4)')
    parser.add_argument('--hyp', type=str, default=None,
                       help='Custom hyperparameters YAML (default: hyp_fish_ultra_stable.yaml)')
    parser.add_argument('--name', type=str, required=True,
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Validate batch size
    if args.batch < 32:
        print(f"⚠ WARNING: Batch size {args.batch} is small. Recommend >= 48 for stability.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Train
    train_ultra_stable(args)


if __name__ == '__main__':
    main()
