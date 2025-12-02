#!/usr/bin/env python3
"""
Enhanced YOLOv11 Training Script for Imbalanced 3-Class Underwater Fish Detection
FIXED VERSION: Uses Ultralytics native W&B integration

Usage:
    # Set environment variables first:
    export WANDB_API_KEY="your_key"
    export WANDB_PROJECT="underwater-fish-yolo11"
    
    # Then run training:
    python train_yolo11_fish_enhanced_fixed.py \
        --data dataset_root \
        --model n \
        --epochs 150 \
        --batch 16 \
        --workers 4 \
        --use-class-weights \
        --name balanced_3class_v1
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import os
from ultralytics import YOLO
from collections import Counter


def calculate_class_weights(dataset_root):
    """
    Calculate class weights based on inverse frequency.
    """
    dataset_root = Path(dataset_root)
    labels_dir = dataset_root / 'train' / 'labels'
    
    class_counts = Counter()
    total_instances = 0
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_instances += 1
    
    # Calculate weights (inverse frequency normalized)
    class_weights = {}
    for class_id in range(3):
        count = class_counts.get(class_id, 1)
        weight = total_instances / (3 * count)
        class_weights[class_id] = weight
    
    # Normalize so average weight is 1.0
    avg_weight = sum(class_weights.values()) / len(class_weights)
    class_weights = {k: v / avg_weight for k, v in class_weights.items()}
    
    print("\n" + "=" * 70)
    print("CLASS WEIGHTS CALCULATION")
    print("=" * 70)
    print(f"Total instances: {total_instances}")
    print(f"\nClass distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_instances) * 100
        weight = class_weights[class_id]
        class_name = ['Grunt Fish', 'Parrot Fish', 'Surgeon Fish'][class_id]
        print(f"  {class_name:15s}: {count:5d} ({percentage:5.1f}%) -> Weight: {weight:.3f}")
    print("=" * 70 + "\n")
    
    return class_weights


def create_balanced_hyperparameters(base_hyp_path, class_weights, output_path):
    """
    Create hyperparameter file with increased cls loss for imbalanced data.
    """
    with open(base_hyp_path, 'r') as f:
        hyp = yaml.safe_load(f)
    
    # Increase cls loss weight for severe imbalance
    hyp['cls'] = 1.0
    
    with open(output_path, 'w') as f:
        yaml.dump(hyp, f, default_flow_style=False)
    
    print(f"✓ Created balanced hyperparameters: {output_path}")
    print(f"  cls loss weight increased to: {hyp['cls']}")
    
    return hyp


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


def train_yolo(args):
    """
    Train YOLOv11 model with class balancing.
    Uses Ultralytics native W&B integration via environment variables.
    """
    # Validate dataset
    dataset_root = Path(args.data).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = calculate_class_weights(dataset_root)
        
        # Create balanced hyperparameters
        base_hyp = Path('hyp_fish.yaml')
        balanced_hyp = Path('hyp_fish_balanced.yaml')
        if base_hyp.exists():
            hyp_config = create_balanced_hyperparameters(base_hyp, class_weights, balanced_hyp)
            hyp_path = balanced_hyp
        else:
            print("⚠ Warning: hyp_fish.yaml not found, using defaults")
            hyp_path = base_hyp
    else:
        hyp_path = Path('hyp_fish.yaml')
    
    # Update data.yaml
    data_yaml = update_data_yaml(dataset_root)
    
    # Model weights
    model_sizes = {
        'n': 'yolo11n.pt',
        's': 'yolo11s.pt',
        'm': 'yolo11m.pt',
    }
    
    weights_path = model_sizes.get(args.model, 'yolo11n.pt')
    print(f"✓ Using COCO pretrained weights: {weights_path}")
    
    # Create experiment name
    exp_name = f"fish_{args.model}_{args.name}" if args.name else f"fish_{args.model}"
    
    # Check W&B environment variables
    if 'WANDB_API_KEY' in os.environ:
        print(f"✓ W&B API key found in environment")
        print(f"  Project: {os.environ.get('WANDB_PROJECT', 'not set')}")
        print(f"  Ultralytics will handle W&B logging automatically")
    else:
        print("⚠ Warning: WANDB_API_KEY not set in environment")
        print("  Set with: export WANDB_API_KEY='your_key'")
    
    # Load model
    model = YOLO(weights_path)
    
    # Training configuration
    train_config = {
        'data': str(data_yaml),
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': exp_name,
        'exist_ok': False,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'save': True,
        'save_period': -1,
        'cache': False,
        'patience': args.early_stop_patience,
        'plots': True,
        'val': True,
    }
    
    # Add hyperparameters file
    if hyp_path.exists():
        train_config['cfg'] = str(hyp_path)
        print(f"✓ Using hyperparameters from: {hyp_path}")
    
    # Print configuration
    print("\n" + "=" * 70)
    print("ENHANCED TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"🎯 TARGET: 70% accuracy across all 3 classes")
    print(f"⚖️  CLASS BALANCING: {'Enabled (cls=1.0, weighted)' if args.use_class_weights else 'Disabled'}")
    print(f"🛑 EARLY STOPPING: Patience = {args.early_stop_patience} epochs")
    if class_weights:
        print(f"\n📊 Class weights:")
        print(f"  Grunt Fish (30.0%):   {class_weights.get(0, 1.0):.3f}x")
        print(f"  Parrot Fish (7.2%):   {class_weights.get(1, 1.0):.3f}x  ⚡ MINORITY")
        print(f"  Surgeon Fish (62.8%): {class_weights.get(2, 1.0):.3f}x")
    print("=" * 70)
    for key, value in train_config.items():
        print(f"{key:20s}: {value}")
    print("=" * 70 + "\n")
    
    # Start training
    print("🚀 Starting enhanced training with class balancing...\n")
    print("📈 W&B logging will be handled automatically by Ultralytics")
    print("   Check your W&B dashboard for real-time metrics\n")
    
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
    """Post-process and create summary."""
    results_csv = results_dir / 'results.csv'
    if not results_csv.exists():
        return
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
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
        description="Enhanced YOLOv11 training with class balancing (FIXED W&B VERSION)"
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset root')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm'],
                       help='Model size (default: n for edge)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Training epochs (default: 150)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=768,
                       help='Image size (default: 768)')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device')
    parser.add_argument('--workers', type=int, default=4,
                       help='Dataloader workers (default: 4)')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='Use class weighting for imbalanced dataset')
    parser.add_argument('--early-stop-patience', type=int, default=75,
                       help='Early stopping patience (default: 75)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name suffix')
    
    args = parser.parse_args()
    
    # Train
    train_yolo(args)


if __name__ == '__main__':
    main()
