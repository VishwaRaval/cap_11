#!/usr/bin/env python3
"""
YOLOv11 Training with Native Resolution Support
Properly handles 768×432 images without stretching to 768×768
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


def train_native_resolution(args):
    """
    Train YOLOv11 with native image resolution (no stretching).
    """
    # Validate paths
    dataset_root = Path(args.data).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")
    
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    # Set up output directory
    output_base = Path("/scratch/am14419/projects/cap_11/runs/detect").resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Output directory: {output_base}")
    
    # Update data.yaml
    data_yaml = update_data_yaml(dataset_root)
    
    # Load model
    print(f"\n✓ Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    # Hyperparameters path
    hyp_path = Path(args.hyp).resolve() if args.hyp else None
    if hyp_path and not hyp_path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {hyp_path}")
    
    if hyp_path:
        print(f"✓ Using hyperparameters: {hyp_path}")
    
    # Handle image size
    if len(args.imgsz) == 1:
        imgsz = args.imgsz[0]
        img_config = f"{imgsz}×{imgsz} (square)"
    elif len(args.imgsz) == 2:
        imgsz = args.imgsz  # [width, height]
        img_config = f"{imgsz[0]}×{imgsz[1]} (native aspect ratio)"
    else:
        raise ValueError("imgsz must be 1 or 2 values")
    
    # Training configuration
    train_config = {
        # Data
        'data': str(data_yaml),
        
        # Training schedule
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': imgsz,  # Can be int or [width, height]
        
        # Hardware
        'device': args.device,
        'workers': args.workers,
        'cache': False,
        
        # Output paths (ABSOLUTE)
        'project': str(output_base),
        'name': args.name,
        'exist_ok': False,
        
        # Optimizer
        'optimizer': 'AdamW',
        
        # Learning rate schedule
        'cos_lr': True,  # Cosine scheduling
        
        # Early stopping
        'patience': args.patience,
        
        # Augmentation control
        'close_mosaic': 10,
        
        # Validation
        'val': True,
        'plots': True,
        'save': True,
        'save_period': -1,  # Only save best
        
        # Stability
        'deterministic': True,
        'seed': 42,
        'verbose': True,
        'amp': True,
        
        # Model settings
        'pretrained': False,
        'single_cls': False,
        'rect': False,  # Don't use rect mode (we're setting exact size)
        'resume': False,
        'freeze': None,
    }
    
    # Add hyperparameters file
    if hyp_path:
        train_config['cfg'] = str(hyp_path)
    
    # Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION - NATIVE RESOLUTION")
    print("=" * 80)
    print(f"\n🎯 GOAL: Train at native resolution without stretching/padding")
    print(f"\n📐 IMAGE RESOLUTION:")
    print(f"  Original images: 768×432")
    print(f"  Training size:   {img_config}")
    if len(args.imgsz) == 2:
        print(f"  ✅ No stretching - preserves aspect ratio!")
    else:
        print(f"  ⚠️  Square format - will add padding/letterboxing")
    
    print(f"\n📊 KEY SETTINGS:")
    print(f"  Batch size: {args.batch}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Patience: {args.patience}")
    print(f"  Optimizer: AdamW")
    print(f"  LR Schedule: Cosine")
    
    print(f"\n📁 PATHS:")
    print(f"  Dataset: {dataset_root}")
    print(f"  Base model: {model_path}")
    print(f"  Output: {output_base / args.name}")
    
    print("\n" + "=" * 80)
    
    # Confirmation if using native resolution
    if len(args.imgsz) == 2:
        print("\n✅ TRAINING WITH NATIVE ASPECT RATIO")
        print("   This should give better results than 768×768 square!")
        print("   Images will NOT be stretched or have excess padding.\n")
    
    # Start training
    print("🚀 Starting training...\n")
    
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
    
    # Print final metrics
    if len(summary_df) > 0:
        final = summary_df.iloc[-1]
        best_idx = summary_df['avg_accuracy'].idxmax() if 'avg_accuracy' in summary_df else -1
        best = summary_df.iloc[best_idx] if best_idx >= 0 else final
        
        print("\n" + "=" * 80)
        print("📈 FINAL METRICS")
        print("=" * 80)
        
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


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 at native resolution (768×432) without stretching"
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset root')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to .pt weights for fine-tuning')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Maximum training epochs (default: 150)')
    parser.add_argument('--batch', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device (default: 0)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Dataloader workers (default: 4)')
    parser.add_argument('--hyp', type=str, default=None,
                       help='Custom hyperparameters YAML')
    parser.add_argument('--name', type=str, required=True,
                       help='Experiment name')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[768, 432],
                       help='Image size: single value (square) or [width, height] (default: [768, 432] native)')
    
    args = parser.parse_args()
    
    # Validate image size
    if len(args.imgsz) not in [1, 2]:
        print("❌ Error: --imgsz must be 1 value (square) or 2 values [width, height]")
        return
    
    if len(args.imgsz) == 2:
        print(f"\n✅ Training at NATIVE resolution: {args.imgsz[0]}×{args.imgsz[1]}")
        print("   This preserves aspect ratio without stretching!")
    else:
        print(f"\n⚠️  Training at SQUARE resolution: {args.imgsz[0]}×{args.imgsz[0]}")
        print("   Images will be letterboxed (padded)")
    
    # Train
    train_native_resolution(args)


if __name__ == '__main__':
    main()
