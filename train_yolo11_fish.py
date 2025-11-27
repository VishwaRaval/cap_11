#!/usr/bin/env python3
"""
YOLOv11 Training Script for Underwater Fish Detection with W&B Integration

Supports:
- Multiple model sizes (nano, small)
- COCO pretrained or custom checkpoint initialization
- Detailed per-epoch metrics logging
- Weights & Biases experiment tracking
- Edge deployment focused on YOLOv11n

Usage:
    # Train YOLOv11n from COCO weights with W&B
    python train_yolo11_fish.py --data dataset_root_preprocessed --model n --epochs 100 --batch 16

    # Train without W&B
    python train_yolo11_fish.py --data dataset_root --model n --epochs 100 --batch 16 --no-wandb

    # Train from custom checkpoint
    python train_yolo11_fish.py --data dataset_root --model n --epochs 100 --batch 16 --weights custom_checkpoint.pt

    # Train YOLOv11s for comparison
    python train_yolo11_fish.py --data dataset_root --model s --epochs 100 --batch 8
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import shutil
import os
from ultralytics import YOLO

# W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ Warning: wandb not installed. Install with: pip install wandb")


def setup_wandb(args, train_config):
    """
    Initialize Weights & Biases tracking.
    
    Args:
        args: Command line arguments
        train_config: Training configuration dictionary
    
    Returns:
        wandb run object or None if disabled
    """
    if not WANDB_AVAILABLE or args.no_wandb:
        return None
    
    # Login to wandb (will use environment variable or cached login)
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    else:
        # Try to login with cached credentials or environment variable
        try:
            wandb.login()
        except Exception as e:
            print(f"⚠ Warning: W&B login failed: {e}")
            print("  Set WANDB_API_KEY environment variable or use --wandb-key")
            return None
    
    # Experiment name
    exp_name = f"fish_{args.model}_{args.name}" if args.name else f"fish_{args.model}"
    
    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        name=exp_name,
        config={
            "model_size": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "image_size": args.imgsz,
            "dataset": str(args.data),
            "optimizer": train_config.get('optimizer', 'auto'),
            "device": args.device,
            "architecture": f"YOLOv11{args.model}",
            "task": "underwater_fish_detection",
            "target_deployment": "edge_device",
            "size_constraint_mb": 70,
        },
        tags=["yolov11", "underwater", "fish-detection", "edge-deployment", args.model],
        notes=args.wandb_notes,
    )
    
    print(f"✓ W&B initialized: {run.name}")
    print(f"  Project: {args.wandb_project}")
    print(f"  URL: {run.url}")
    
    return run


def log_to_wandb(results_dir, wandb_run):
    """
    Log training results to Weights & Biases.
    
    Args:
        results_dir: Path to results directory
        wandb_run: Active wandb run object
    """
    if wandb_run is None:
        return
    
    # Log results CSV
    results_csv = results_dir / 'results.csv'
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        # Log each epoch
        for idx, row in df.iterrows():
            metrics = {}
            
            # Training losses
            if 'train/box_loss' in row:
                metrics['train/box_loss'] = row['train/box_loss']
            if 'train/cls_loss' in row:
                metrics['train/cls_loss'] = row['train/cls_loss']
            if 'train/dfl_loss' in row:
                metrics['train/dfl_loss'] = row['train/dfl_loss']
            
            # Validation losses
            if 'val/box_loss' in row:
                metrics['val/box_loss'] = row['val/box_loss']
            if 'val/cls_loss' in row:
                metrics['val/cls_loss'] = row['val/cls_loss']
            if 'val/dfl_loss' in row:
                metrics['val/dfl_loss'] = row['val/dfl_loss']
            
            # Performance metrics
            if 'metrics/precision(B)' in row:
                metrics['precision'] = row['metrics/precision(B)']
            if 'metrics/recall(B)' in row:
                metrics['recall'] = row['metrics/recall(B)']
            if 'metrics/mAP50(B)' in row:
                metrics['mAP50'] = row['metrics/mAP50(B)']
            if 'metrics/mAP50-95(B)' in row:
                metrics['mAP50-95'] = row['metrics/mAP50-95(B)']
            
            # Log to wandb
            wandb_run.log(metrics, step=int(row.get('epoch', idx)))
    
    # Log training curves as images
    for plot_file in results_dir.glob('*.png'):
        try:
            wandb_run.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
        except Exception as e:
            print(f"⚠ Warning: Could not log {plot_file.name} to W&B: {e}")
    
    # Log best model weights as artifact
    best_weights = results_dir / 'weights' / 'best.pt'
    if best_weights.exists():
        try:
            artifact = wandb.Artifact(
                name=f"model-{wandb_run.name}",
                type="model",
                description=f"Best YOLOv11{args.model} weights for underwater fish detection"
            )
            artifact.add_file(str(best_weights))
            wandb_run.log_artifact(artifact)
            print(f"✓ Logged best weights to W&B as artifact")
        except Exception as e:
            print(f"⚠ Warning: Could not log weights to W&B: {e}")
    
    # Log final metrics summary
    summary_csv = results_dir / 'metrics_summary.csv'
    if summary_csv.exists():
        try:
            summary_df = pd.read_csv(summary_csv)
            if len(summary_df) > 0:
                final_metrics = summary_df.iloc[-1].to_dict()
                wandb_run.summary.update({
                    f"final_{k}": v for k, v in final_metrics.items() 
                    if k != 'epoch' and pd.notna(v)
                })
        except Exception as e:
            print(f"⚠ Warning: Could not log summary to W&B: {e}")


def update_data_yaml(dataset_root):
    """
    Update data.yaml to use absolute paths and verify structure.
    
    Args:
        dataset_root: Path to dataset root directory
    
    Returns:
        Path to updated data.yaml
    """
    dataset_root = Path(dataset_root).resolve()
    data_yaml_path = dataset_root / 'data.yaml'
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_root}")
    
    # Read existing data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Update paths to absolute
    data_config['path'] = str(dataset_root)
    data_config['train'] = 'train/images'
    data_config['val'] = 'valid/images'
    data_config['test'] = 'test/images'
    
    # Ensure nc (number of classes) is set
    if 'nc' not in data_config:
        # Count classes from names
        data_config['nc'] = len(data_config.get('names', []))
    
    # Write updated data.yaml
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✓ Updated data.yaml with absolute paths")
    print(f"  Dataset root: {dataset_root}")
    print(f"  Classes: {data_config['nc']}")
    print(f"  Class names: {data_config.get('names', [])}")
    
    return data_yaml_path


def train_yolo(args):
    """
    Train YOLOv11 model with specified configuration.
    
    Args:
        args: Command line arguments
    """
    # Validate dataset
    dataset_root = Path(args.data).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    # Update data.yaml
    data_yaml = update_data_yaml(dataset_root)
    
    # Determine model weights
    model_sizes = {
        'n': 'yolo11n.pt',  # Nano - for edge deployment
        's': 'yolo11s.pt',  # Small - for comparison
        'm': 'yolo11m.pt',  # Medium - if you have more compute
    }
    
    if args.weights and Path(args.weights).exists():
        # Use custom weights (e.g., Roboflow checkpoint)
        weights_path = args.weights
        print(f"✓ Using custom weights: {weights_path}")
    else:
        # Use COCO pretrained weights
        weights_path = model_sizes.get(args.model, 'yolo11n.pt')
        print(f"✓ Using COCO pretrained weights: {weights_path}")
    
    # Create experiment name
    exp_name = f"fish_{args.model}_{args.name}" if args.name else f"fish_{args.model}"
    
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
        'optimizer': 'auto',  # Auto-select SGD or Adam
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,  # Rectangular training disabled
        'cos_lr': False,  # Cosine LR scheduler
        'close_mosaic': 10,  # Disable mosaic last 10 epochs
        'resume': False,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,  # Use 100% of dataset
        'profile': False,
        'freeze': None,  # No layer freezing
        'save': True,
        'save_period': -1,  # Save checkpoint every N epochs (-1 = only last)
        'cache': False,  # Don't cache images (save RAM)
        'patience': 50,  # Early stopping patience
        'plots': True,
        'val': True,
    }
    
    # Add hyperparameters file if exists
    hyp_path = Path('hyp_fish.yaml')
    if hyp_path.exists():
        train_config['cfg'] = str(hyp_path)
        print(f"✓ Using hyperparameters from: {hyp_path}")
    else:
        print("⚠ Warning: hyp_fish.yaml not found, using default hyperparameters")
    
    # Initialize W&B
    wandb_run = setup_wandb(args, train_config)
    
    # Print training configuration
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    for key, value in train_config.items():
        print(f"{key:20s}: {value}")
    print("=" * 70 + "\n")
    
    # Start training
    print("🚀 Starting training...\n")
    results = model.train(**train_config)
    
    # Training complete
    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)
    
    # Find results directory
    results_dir = Path(args.project) / exp_name
    
    # Post-process metrics
    post_process_metrics(results_dir)
    
    # Log to W&B
    if wandb_run is not None:
        print("\n📊 Logging results to Weights & Biases...")
        log_to_wandb(results_dir, wandb_run)
        wandb_run.finish()
        print("✓ W&B logging complete")
    
    print(f"\n📊 Results saved to: {results_dir}")
    print(f"📁 Best weights: {results_dir / 'weights' / 'best.pt'}")
    print(f"📁 Last weights: {results_dir / 'weights' / 'last.pt'}")
    print(f"📁 Metrics CSV: {results_dir / 'results.csv'}")
    
    return results_dir


def post_process_metrics(results_dir):
    """
    Post-process and enhance metrics CSV from Ultralytics.
    Creates a cleaner summary with key metrics.
    
    Args:
        results_dir: Path to results directory
    """
    results_csv = results_dir / 'results.csv'
    
    if not results_csv.exists():
        print(f"⚠ Warning: results.csv not found in {results_dir}")
        return
    
    # Read results
    df = pd.read_csv(results_csv)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Create enhanced summary
    summary_cols = []
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
    
    # Build summary dataframe
    summary_data = {}
    for orig_col, new_col in col_mapping.items():
        if orig_col in df.columns:
            summary_data[new_col] = df[orig_col]
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save enhanced summary
        summary_csv = results_dir / 'metrics_summary.csv'
        summary_df.to_csv(summary_csv, index=False, float_format='%.4f')
        
        print(f"\n✓ Created enhanced metrics summary: {summary_csv}")
        
        # Print final epoch metrics
        if len(summary_df) > 0:
            final = summary_df.iloc[-1]
            print("\n📈 Final Epoch Metrics:")
            print("-" * 50)
            for col in summary_df.columns:
                if col != 'epoch':
                    print(f"  {col:20s}: {final[col]:.4f}")
            print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 for underwater fish detection with W&B tracking"
    )
    
    # Dataset arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset root directory')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm'],
                       help='Model size: n (nano), s (small), m (medium). Default: n for edge deployment')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to custom weights (e.g., Roboflow checkpoint). If not provided, uses COCO pretrained')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16 for nano, 8 for small recommended)')
    parser.add_argument('--imgsz', type=int, default=768,
                       help='Input image size (default: 768 to match Roboflow preprocessing)')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device, i.e. 0 or 0,1,2,3 or cpu (default: 0)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of dataloader workers (default: 8)')
    
    # Output arguments
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory (default: runs/detect)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name suffix (default: fish_{model})')
    
    # W&B arguments
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='underwater-fish-detection',
                       help='W&B project name (default: underwater-fish-detection)')
    parser.add_argument('--wandb-key', type=str, default=None,
                       help='W&B API key (or set WANDB_API_KEY environment variable)')
    parser.add_argument('--wandb-notes', type=str, default=None,
                       help='Notes for this W&B run')
    
    args = parser.parse_args()
    
    # Train model
    train_yolo(args)


if __name__ == '__main__':
    main()