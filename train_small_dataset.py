#!/usr/bin/env python3
"""
YOLOv11 Training Script - OPTIMIZED FOR SMALL DATASETS
Prevents overfitting on limited data (~2950 images)

Key strategies:
- Very low learning rate (0.0005)
- Large batch size (32-64) for stable gradients and fewer updates
- Strong regularization (dropout, weight decay)
- Minimal augmentation
- Longer training with early stopping

Usage:
    python train_small_dataset.py \
        --data /path/to/dataset \
        --model s \
        --batch 32 \
        --epochs 200 \
        --name small_dataset_conservative_v1
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


def train_yolo_small_dataset(args):
    """
    Train YOLOv11 optimized for small datasets.
    """
    # Validate dataset
    dataset_root = Path(args.data).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    # Update data.yaml
    data_yaml = update_data_yaml(dataset_root)
    
    # Model weights
    if args.model.endswith('.pt'):
        weights_path = args.model
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        print(f"✓ Loading existing weights for fine-tuning: {weights_path}")
        
        # Extract model size from path
        model_size = 's'
        if 'fish_m_' in str(weights_path) or '/m_' in str(weights_path):
            model_size = 'm'
        elif 'fish_n_' in str(weights_path) or '/n_' in str(weights_path):
            model_size = 'n'
        elif 'fish_s_' in str(weights_path) or '/s_' in str(weights_path):
            model_size = 's'
    else:
        model_sizes = {
            'n': 'yolo11n.pt',
            's': 'yolo11s.pt',
            'm': 'yolo11m.pt',
        }
        model_size = args.model
        weights_path = model_sizes.get(args.model, 'yolo11s.pt')
        print(f"✓ Using COCO pretrained weights: {weights_path}")
    
    # Experiment name
    if args.name:
        exp_name = f"fish_{model_size}_{args.name}"
    else:
        exp_name = f"fish_{model_size}_small_dataset"
    
    # Load model
    model = YOLO(weights_path)
    
    # Calculate effective updates per epoch
    train_images = len(list((dataset_root / 'train' / 'images').glob('*.jpg'))) + \
                   len(list((dataset_root / 'train' / 'images').glob('*.png')))
    updates_per_epoch = train_images // args.batch
    total_updates = updates_per_epoch * args.epochs
    
    print("\n" + "=" * 70)
    print("SMALL DATASET TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"🎯 STRATEGY: Prevent overfitting on limited data")
    print(f"📊 Dataset size: {train_images} images")
    print(f"📦 Batch size: {args.batch} (LARGE for stable gradients)")
    print(f"🔄 Updates per epoch: {updates_per_epoch}")
    print(f"🔄 Total updates: {total_updates:,} ({args.epochs} epochs)")
    print(f"📉 Learning rate: 0.0005 (VERY LOW)")
    print(f"🛡️  Regularization: Strong (dropout=0.2, weight_decay=0.001)")
    print(f"🎨 Augmentation: Minimal (preserve limited data)")
    print("=" * 70)
    
    # Training configuration - OPTIMIZED FOR SMALL DATASETS
    train_config = {
        'data': str(data_yaml),
        'epochs': args.epochs,
        'batch': args.batch,           # LARGE batch size
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': exp_name,
        'exist_ok': False,
        'pretrained': True,
        'optimizer': 'AdamW',          # ⬆️ AdamW better for small datasets
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,               # ⬇️ Disable cosine (use linear for stability)
        'close_mosaic': 0,             # ⬇️ No mosaic anyway
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'save': True,
        'save_period': -1,
        'cache': True,                 # ⬆️ Cache for faster training
        'patience': args.patience,     # ⬆️ Higher patience for small datasets
        'plots': True,
        'val': True,
    }
    
    # Hyperparameters
    hyp_path = Path(args.hyp) if args.hyp else Path('hyp_fish_small_dataset.yaml')
    
    if hyp_path.exists():
        train_config['cfg'] = str(hyp_path)
        print(f"\n✓ Using hyperparameters from: {hyp_path}")
        
        # Print key hyperparameters
        with open(hyp_path, 'r') as f:
            hyp = yaml.safe_load(f)
        print(f"\n🔧 Key Hyperparameters:")
        print(f"  lr0:           {hyp.get('lr0', 'N/A')} (initial learning rate)")
        print(f"  weight_decay:  {hyp.get('weight_decay', 'N/A')} (L2 regularization)")
        print(f"  dropout:       {hyp.get('dropout', 'N/A')} (dropout rate)")
        print(f"  mosaic:        {hyp.get('mosaic', 'N/A')} (mosaic augmentation)")
        print(f"  mixup:         {hyp.get('mixup', 'N/A')} (mixup augmentation)")
    else:
        print(f"⚠️  Warning: {hyp_path} not found, using Ultralytics defaults")
    
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION DETAILS")
    print("=" * 70)
    for key, value in train_config.items():
        print(f"{key:20s}: {value}")
    print("=" * 70 + "\n")
    
    # Check W&B
    if 'WANDB_API_KEY' in os.environ:
        print(f"✓ W&B logging enabled")
        print(f"  Project: {os.environ.get('WANDB_PROJECT', 'yolo11')}")
    
    # Start training
    print("\n🚀 Starting training optimized for small dataset...\n")
    print("💡 Strategy:")
    print("  • Large batch (stable gradients, fewer updates)")
    print("  • Very low LR (prevent overfitting)")
    print("  • Strong regularization (dropout + weight decay)")
    print("  • Minimal augmentation (preserve data)")
    print("  • High patience (wait for convergence)")
    print()
    
    results = model.train(**train_config)
    
    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)
    
    # Find results directory
    results_dir = Path(args.project) / exp_name
    
    # Post-process metrics
    post_process_metrics(results_dir)
    
    print(f"\n📊 Results saved to: {results_dir}")
    print(f"📁 Best weights: {results_dir / 'weights' / 'best.pt'}")
    
    return results_dir


def post_process_metrics(results_dir):
    """Post-process and create summary with overfitting detection."""
    results_csv = results_dir / 'results.csv'
    if not results_csv.exists():
        return
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # Detect overfitting
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        train_loss = df['train/box_loss'].values
        val_loss = df['val/box_loss'].values
        
        # Check if validation loss diverging
        if len(train_loss) > 20:
            early_val = val_loss[10:20].mean()
            late_val = val_loss[-10:].mean()
            
            print(f"\n🔍 Overfitting Analysis:")
            print(f"  Early validation loss (epochs 10-20): {early_val:.4f}")
            print(f"  Late validation loss (last 10 epochs): {late_val:.4f}")
            
            if late_val > early_val * 1.1:
                print(f"  ⚠️  WARNING: Validation loss increased by {((late_val/early_val - 1)*100):.1f}%")
                print(f"     Model may be overfitting!")
            else:
                print(f"  ✅ Validation loss stable/decreasing - good!")
    
    # Create summary
    col_mapping = {
        'epoch': 'epoch',
        'train/box_loss': 'train_box_loss',
        'train/cls_loss': 'train_cls_loss',
        'val/box_loss': 'val_box_loss',
        'val/cls_loss': 'val_cls_loss',
        'metrics/precision(B)': 'precision',
        'metrics/recall(B)': 'recall',
        'metrics/mAP50(B)': 'mAP50',
        'metrics/mAP50-95(B)': 'mAP50_95',
    }
    
    summary_data = {}
    for orig_col, new_col in col_mapping.items():
        if orig_col in df.columns:
            summary_data[new_col] = df[orig_col]
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        if 'precision' in summary_df.columns and 'recall' in summary_df.columns:
            summary_df['f1_score'] = 2 * (summary_df['precision'] * summary_df['recall']) / (summary_df['precision'] + summary_df['recall'])
            summary_df['avg_accuracy'] = (summary_df['precision'] + summary_df['recall']) / 2
            summary_df['gap_to_70pct'] = 0.70 - summary_df['avg_accuracy']
        
        summary_csv = results_dir / 'metrics_summary.csv'
        summary_df.to_csv(summary_csv, index=False, float_format='%.4f')
        
        print(f"\n✓ Created metrics summary: {summary_csv}")
        
        if len(summary_df) > 0:
            final = summary_df.iloc[-1]
            print("\n📈 Final Epoch Metrics:")
            print("-" * 50)
            for col in summary_df.columns:
                if col != 'epoch':
                    value = final[col]
                    if pd.notna(value):
                        print(f"  {col:25s}: {value:.4f}")
            print("-" * 50)
            
            # Check if target met
            if 'avg_accuracy' in final:
                avg_acc = final['avg_accuracy']
                print(f"\n🎯 Target Achievement:")
                print(f"  Average Accuracy: {avg_acc:.1%}")
                print(f"  Target (70%): {'✅ MET' if avg_acc >= 0.70 else f'❌ GAP: {(0.70-avg_acc)*100:.1f}%'}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv11 training optimized for small datasets"
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset root')
    parser.add_argument('--model', type=str, default='s',
                       help='Model size (n/s/m) or path to .pt weights')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Training epochs (default: 200 for small datasets)')
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size (default: 32, use 48-64 if GPU allows)')
    parser.add_argument('--imgsz', type=int, default=768,
                       help='Image size (default: 768)')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device')
    parser.add_argument('--workers', type=int, default=4,
                       help='Dataloader workers (default: 4)')
    parser.add_argument('--hyp', type=str, default='hyp_fish_small_dataset.yaml',
                       help='Path to hyperparameters YAML')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience (default: 100 for small datasets)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name suffix')
    
    args = parser.parse_args()
    
    # Validate batch size
    if args.batch < 16:
        print(f"⚠️  WARNING: Batch size {args.batch} is very small!")
        print(f"   For small datasets, larger batches (32-64) give more stable gradients")
        print(f"   and fewer parameter updates (helps prevent overfitting)")
    
    # Train
    train_yolo_small_dataset(args)


if __name__ == '__main__':
    main()
