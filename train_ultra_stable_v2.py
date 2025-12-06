#!/usr/bin/env python3
"""
ULTRA-STABLE YOLOv11 Training with Multiple Checkpoint Saving

Strategy:
- Large batch size for stable gradients
- Very low learning rate for minimal updates
- AdamW optimizer for better convergence
- Saves 4 checkpoints: best.pt (mAP50), best_prec.pt, best_rec.pt, last.pt

Usage:
    python train_ultra_stable_v2.py \
        --data /scratch/am14419/projects/cap_11/dataset_root \
        --model yolo11m.pt \
        --batch 64 \
        --name medium_precision_v1_scratch
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import os
import shutil
import torch
from ultralytics import YOLO


class MultiCheckpointTracker:
    """Track and save multiple checkpoints based on different metrics"""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.weights_dir = self.save_dir / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best values
        self.best_precision = 0.0
        self.best_recall = 0.0
        self.best_map50 = 0.0
        
        # Tracking file
        self.tracker_file = self.weights_dir / 'checkpoint_tracker.txt'
        
        print(f"\n✓ Multi-checkpoint tracker initialized")
        print(f"  Will save: best.pt, best_prec.pt, best_rec.pt, last.pt")
    
    def on_fit_epoch_end(self, trainer):
        """Called at the end of each training epoch"""
        # Get current metrics
        metrics = trainer.metrics
        
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        map50 = metrics.get('metrics/mAP50(B)', 0)
        epoch = trainer.epoch
        
        # Paths
        best_prec_path = self.weights_dir / 'best_prec.pt'
        best_rec_path = self.weights_dir / 'best_rec.pt'
        
        # Helper function to save checkpoint
        def save_checkpoint(path, metric_value, metric_name):
            """Save model checkpoint in YOLO format"""
            try:
                # Get the model state - use EMA if available (it's better)
                model_to_save = trainer.ema.ema if trainer.ema else trainer.model
                
                # Create checkpoint dict in YOLO format
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': metric_value,
                    'model': model_to_save.state_dict(),
                    'train_args': vars(trainer.args),
                    'date': None,
                    'version': None,
                }
                
                # Save checkpoint
                torch.save(ckpt, path)
                print(f"  📊 New best {metric_name}: {metric_value:.4f} (epoch {epoch}) → saved to {path.name}")
                return True
            except Exception as e:
                print(f"  ⚠️  Failed to save {path.name}: {e}")
                return False
        
        # Check if we have new best precision
        if precision > self.best_precision:
            self.best_precision = precision
            save_checkpoint(best_prec_path, precision, 'precision')
        
        # Check if we have new best recall
        if recall > self.best_recall:
            self.best_recall = recall
            save_checkpoint(best_rec_path, recall, 'recall')
        
        # Track mAP50 (YOLO does this by default, but we track for logging)
        if map50 > self.best_map50:
            self.best_map50 = map50
            print(f"  📊 New best mAP50: {map50:.4f} (epoch {epoch}) → YOLO saves to best.pt")
        
        # Save tracking info
        with open(self.tracker_file, 'w') as f:
            f.write(f"Best Precision: {self.best_precision:.4f}\n")
            f.write(f"Best Recall: {self.best_recall:.4f}\n")
            f.write(f"Best mAP50: {self.best_map50:.4f}\n")


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
    Train YOLOv11 with ultra-stable settings and multi-checkpoint saving.
    """
    # Validate paths
    dataset_root = Path(args.data).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")
    
    # Handle model path (could be pretrained or checkpoint)
    model_path = args.model
    if not model_path.endswith('.pt'):
        model_path = model_path + '.pt'
    
    # Check if it's a local file or a model name
    if Path(model_path).exists():
        model_path = str(Path(model_path).resolve())
        print(f"✓ Using local model: {model_path}")
    else:
        # Assume it's a pretrained model name (e.g., 'yolo11m.pt')
        print(f"✓ Using pretrained model: {model_path}")
    
    # Set up output directory with ABSOLUTE PATH
    output_base = Path("/scratch/am14419/projects/cap_11/runs/detect").resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Output directory: {output_base}")
    
    # Update data.yaml
    data_yaml = update_data_yaml(dataset_root)
    
    # Load model
    print(f"\n✓ Loading model...")
    model = YOLO(model_path)
    
    # Hyperparameters path
    hyp_path = Path(args.hyp).resolve() if args.hyp else Path("hyp_fish_ultra_stable.yaml").resolve()
    if not hyp_path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {hyp_path}")
    
    print(f"✓ Using hyperparameters: {hyp_path}")
    
    # Determine if pretrained based on model name
    is_pretrained = not any(x in str(model_path).lower() for x in ['runs/', 'detect/', 'train/'])
    
    # Training configuration - ULTRA STABLE
    train_config = {
        # Data
        'data': str(data_yaml),
        
        # Training schedule
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': 768,
        
        # Hardware
        'workers': args.workers,
        'cache': False,  # Don't cache (memory intensive)
        
        # Output paths (ABSOLUTE)
        'project': str(output_base),
        'name': args.name,
        'exist_ok': False,
        
        # Optimizer - AdamW for better convergence
        'optimizer': 'AdamW',
        
        # Learning rate schedule
        'cos_lr': True,  # Cosine scheduling
        
        # Early stopping - AGGRESSIVE
        'patience': args.patience,
        
        # Augmentation control
        'close_mosaic': 0,
        
        # Validation
        'val': True,
        'plots': True,
        'save': True,
        'save_period': -1,  # Only save best (YOLO will save last.pt and best.pt)
        
        # Stability
        'deterministic': True,
        'seed': 42,
        'verbose': True,
        'amp': True,
        
        # Model settings
        'pretrained': is_pretrained,
        'single_cls': False,
        'rect': False,
        'resume': False,
        'freeze': None,
        
        # Hyperparameters file
        'cfg': str(hyp_path),
    }
    
    # Print configuration
    print("\n" + "=" * 80)
    print("ULTRA-STABLE TRAINING WITH MULTI-CHECKPOINT SAVING")
    print("=" * 80)
    print(f"\n🎯 GOAL: Stable training → 70%+ accuracy")
    print(f"\n💾 CHECKPOINT STRATEGY:")
    print(f"  ✓ best.pt       → Best mAP@50 (YOLO default)")
    print(f"  ✓ best_prec.pt  → Best Precision")
    print(f"  ✓ best_rec.pt   → Best Recall")
    print(f"  ✓ last.pt       → Last epoch")
    print(f"\n📊 KEY STRATEGIES:")
    print(f"  ✓ Large batch ({args.batch}) → Stable gradients")
    print(f"  ✓ AdamW optimizer → Better convergence")
    print(f"  ✓ Cosine LR scheduling → Smooth decay")
    print(f"  ✓ Early stopping (patience={args.patience})")
    print(f"\n📁 PATHS:")
    print(f"  Dataset: {dataset_root}")
    print(f"  Base model: {model_path}")
    print(f"  Output: {output_base / args.name}")
    print(f"  Hyperparams: {hyp_path.name}")
    print("=" * 80 + "\n")
    
    # Set up result directory path for checkpoint tracker
    results_dir = output_base / args.name
    
    # Initialize checkpoint tracker
    checkpoint_tracker = MultiCheckpointTracker(results_dir)
    
    # Add callback for multi-checkpoint saving
    def epoch_end_callback(trainer):
        checkpoint_tracker.on_fit_epoch_end(trainer)
    
    # Add the callback
    model.add_callback('on_fit_epoch_end', epoch_end_callback)
    
    # Start training
    print("🚀 Starting ultra-stable training with multi-checkpoint saving...\n")
    
    results = model.train(**train_config)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    
    # Verify all checkpoints exist
    weights_dir = results_dir / 'weights'
    checkpoints = ['best.pt', 'best_prec.pt', 'best_rec.pt', 'last.pt']
    
    print(f"\n💾 SAVED CHECKPOINTS:")
    for ckpt in checkpoints:
        ckpt_path = weights_dir / ckpt
        if ckpt_path.exists():
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {ckpt:15s} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {ckpt:15s} (missing)")
    
    # Post-process metrics
    post_process_metrics(results_dir)
    
    print(f"\n📊 Results saved to: {results_dir}")
    print(f"\n💡 Next steps:")
    print(f"  1. Use compare_models_v2.py to compare all checkpoints")
    print(f"  2. For precision runs: Use best_prec.pt")
    print(f"  3. For recall runs: Use best_rec.pt")
    print(f"  4. For balanced: Use best.pt (best mAP50)")
    
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
        summary_df['f1_score'] = (
            2 * (summary_df['precision'] * summary_df['recall']) / 
            (summary_df['precision'] + summary_df['recall'] + 1e-8)
        )
        summary_df['avg_accuracy'] = (summary_df['precision'] + summary_df['recall']) / 2
        summary_df['gap_to_70pct'] = 0.70 - summary_df['avg_accuracy']
    
    # Save summary
    summary_csv = results_dir / 'metrics_summary.csv'
    summary_df.to_csv(summary_csv, index=False, float_format='%.6f')
    
    print(f"\n✓ Created metrics summary: {summary_csv}")
    
    # Print best metrics for each type
    if len(summary_df) > 0:
        print("\n" + "=" * 80)
        print("📈 BEST METRICS ACROSS ALL EPOCHS")
        print("=" * 80)
        
        if 'precision' in summary_df.columns:
            best_prec_idx = summary_df['precision'].idxmax()
            best_prec = summary_df.iloc[best_prec_idx]
            print(f"\n🎯 Best Precision (Epoch {int(best_prec['epoch'])}):")
            print(f"   Precision: {best_prec['precision']:.2%}")
            print(f"   Recall:    {best_prec['recall']:.2%}")
            print(f"   Avg Acc:   {best_prec['avg_accuracy']:.2%}")
            print(f"   → Saved as best_prec.pt")
        
        if 'recall' in summary_df.columns:
            best_rec_idx = summary_df['recall'].idxmax()
            best_rec = summary_df.iloc[best_rec_idx]
            print(f"\n🎯 Best Recall (Epoch {int(best_rec['epoch'])}):")
            print(f"   Recall:    {best_rec['recall']:.2%}")
            print(f"   Precision: {best_rec['precision']:.2%}")
            print(f"   Avg Acc:   {best_rec['avg_accuracy']:.2%}")
            print(f"   → Saved as best_rec.pt")
        
        if 'mAP50' in summary_df.columns:
            best_map_idx = summary_df['mAP50'].idxmax()
            best_map = summary_df.iloc[best_map_idx]
            print(f"\n🎯 Best mAP@50 (Epoch {int(best_map['epoch'])}):")
            print(f"   mAP@50:    {best_map['mAP50']:.2%}")
            print(f"   Precision: {best_map['precision']:.2%}")
            print(f"   Recall:    {best_map['recall']:.2%}")
            print(f"   Avg Acc:   {best_map['avg_accuracy']:.2%}")
            print(f"   → Saved as best.pt")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-stable YOLOv11 training with multi-checkpoint saving"
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset root')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to .pt weights or model name (e.g., yolo11m.pt)')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Maximum training epochs (default: 300)')
    parser.add_argument('--batch', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Dataloader workers (default: 4)')
    parser.add_argument('--hyp', type=str, default=None,
                       help='Custom hyperparameters YAML')
    parser.add_argument('--name', type=str, required=True,
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Train
    train_ultra_stable(args)


if __name__ == '__main__':
    main()
