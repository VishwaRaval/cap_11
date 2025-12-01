#!/usr/bin/env python3
"""
YOLOv11 Training Script for Underwater Fish Detection with Enhanced W&B Integration

Supports:
- Multiple model sizes (nano, small)
- COCO pretrained or custom checkpoint initialization
- Detailed per-epoch metrics logging with W&B
- Real-time experiment tracking with comprehensive graphing
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
    
    # Load hyperparameters if available
    hyp_config = {}
    hyp_path = Path('hyp_fish.yaml')
    if hyp_path.exists():
        with open(hyp_path, 'r') as f:
            hyp_config = yaml.safe_load(f)
    
    # Build comprehensive config
    config = {
        # Model configuration
        "model_size": args.model,
        "architecture": f"YOLOv11{args.model}",
        "weights_init": "roboflow_transfer" if args.weights else "coco_pretrained",
        
        # Training parameters
        "epochs": args.epochs,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "optimizer": train_config.get('optimizer', 'auto'),
        "learning_rate": hyp_config.get('lr0', 0.01),
        "warmup_epochs": hyp_config.get('warmup_epochs', 3.0),
        "momentum": hyp_config.get('momentum', 0.937),
        "weight_decay": hyp_config.get('weight_decay', 0.0005),
        
        # Loss weights
        "box_loss_gain": hyp_config.get('box', 7.5),
        "cls_loss_gain": hyp_config.get('cls', 0.5),
        "dfl_loss_gain": hyp_config.get('dfl', 1.5),
        
        # Augmentation parameters
        "mosaic": hyp_config.get('mosaic', 1.0),
        "mixup": hyp_config.get('mixup', 0.0),
        "degrees": hyp_config.get('degrees', 5.0),
        "translate": hyp_config.get('translate', 0.1),
        "scale": hyp_config.get('scale', 0.3),
        "shear": hyp_config.get('shear', 2.0),
        "flipud": hyp_config.get('flipud', 0.0),
        "fliplr": hyp_config.get('fliplr', 0.5),
        "hsv_h": hyp_config.get('hsv_h', 0.01),
        "hsv_s": hyp_config.get('hsv_s', 0.4),
        "hsv_v": hyp_config.get('hsv_v', 0.2),
        
        # Dataset info
        "dataset": str(args.data),
        "device": args.device,
        
        # Project constraints
        "task": "underwater_fish_detection",
        "target_deployment": "edge_device",
        "size_constraint_mb": 70,
        "target_recall": 0.60,  # Target improvement from 56.8%
        "current_baseline_recall": 0.568,
        "current_baseline_map50": 0.636,
        "current_baseline_precision": 0.674,
    }
    
    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        name=exp_name,
        config=config,
        tags=["yolov11", "underwater", "fish-detection", "edge-deployment", args.model],
        notes=args.wandb_notes,
    )
    
    # Define custom metrics for better visualization
    # This ensures metrics are tracked as time series graphs
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("metrics/*", step_metric="epoch")
    wandb.define_metric("lr/*", step_metric="epoch")
    
    # Explicitly define key metrics for graphing in the metrics panel
    # These will show up as line graphs with proper min/max summaries
    wandb.define_metric("train/box_loss", step_metric="epoch", summary="min")
    wandb.define_metric("train/cls_loss", step_metric="epoch", summary="min")
    wandb.define_metric("train/dfl_loss", step_metric="epoch", summary="min")
    
    wandb.define_metric("val/box_loss", step_metric="epoch", summary="min")
    wandb.define_metric("val/cls_loss", step_metric="epoch", summary="min")
    wandb.define_metric("val/dfl_loss", step_metric="epoch", summary="min")
    
    wandb.define_metric("metrics/precision(B)", step_metric="epoch", summary="max")
    wandb.define_metric("metrics/recall(B)", step_metric="epoch", summary="max")
    wandb.define_metric("metrics/mAP50(B)", step_metric="epoch", summary="max")
    wandb.define_metric("metrics/mAP50-95(B)", step_metric="epoch", summary="max")
    
    # Learning rates for all parameter groups
    wandb.define_metric("lr/pg0", step_metric="epoch", summary="last")
    wandb.define_metric("lr/pg1", step_metric="epoch", summary="last")
    wandb.define_metric("lr/pg2", step_metric="epoch", summary="last")
    
    print(f"✓ W&B initialized: {run.name}")
    print(f"  Project: {args.wandb_project}")
    print(f"  URL: {run.url}")
    print(f"  📊 Configured metrics for real-time graphing:")
    print(f"     - Training losses: box_loss, cls_loss, dfl_loss")
    print(f"     - Validation losses: box_loss, cls_loss, dfl_loss")
    print(f"     - Performance: Precision, Recall, mAP50, mAP50-95")
    print(f"     - Learning rates: pg0, pg1, pg2")
    
    return run


def log_to_wandb(results_dir, wandb_run, args):
    """
    Log training results to Weights & Biases.
    
    Args:
        results_dir: Path to results directory
        wandb_run: Active wandb run object
        args: Command line arguments
    """
    if wandb_run is None:
        return
    
    # Log results CSV with detailed metrics
    results_csv = results_dir / 'results.csv'
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        print("\n📊 Logging detailed metrics to W&B...")
        
        # Log each epoch with comprehensive metrics
        for idx, row in df.iterrows():
            metrics = {"epoch": int(row.get('epoch', idx))}
            
            # Training losses
            if 'train/box_loss' in row:
                metrics['train/box_loss'] = float(row['train/box_loss'])
            if 'train/cls_loss' in row:
                metrics['train/cls_loss'] = float(row['train/cls_loss'])
            if 'train/dfl_loss' in row:
                metrics['train/dfl_loss'] = float(row['train/dfl_loss'])
            
            # Validation losses
            if 'val/box_loss' in row:
                metrics['val/box_loss'] = float(row['val/box_loss'])
            if 'val/cls_loss' in row:
                metrics['val/cls_loss'] = float(row['val/cls_loss'])
            if 'val/dfl_loss' in row:
                metrics['val/dfl_loss'] = float(row['val/dfl_loss'])
            
            # Performance metrics
            if 'metrics/precision(B)' in row:
                metrics['metrics/precision(B)'] = float(row['metrics/precision(B)'])
            if 'metrics/recall(B)' in row:
                metrics['metrics/recall(B)'] = float(row['metrics/recall(B)'])
            if 'metrics/mAP50(B)' in row:
                metrics['metrics/mAP50(B)'] = float(row['metrics/mAP50(B)'])
            if 'metrics/mAP50-95(B)' in row:
                metrics['metrics/mAP50-95(B)'] = float(row['metrics/mAP50-95(B)'])
            
            # Learning rate (if available)
            if 'lr/pg0' in row:
                metrics['lr/pg0'] = float(row['lr/pg0'])
            if 'lr/pg1' in row:
                metrics['lr/pg1'] = float(row['lr/pg1'])
            if 'lr/pg2' in row:
                metrics['lr/pg2'] = float(row['lr/pg2'])
            
            # Additional metrics if available
            if 'metrics/precision(M)' in row:
                metrics['metrics/mask_precision'] = float(row['metrics/precision(M)'])
            if 'metrics/recall(M)' in row:
                metrics['metrics/mask_recall'] = float(row['metrics/recall(M)'])
            
            # Compute derived metrics
            if 'metrics/precision(B)' in metrics and 'metrics/recall(B)' in metrics:
                p = metrics['metrics/precision(B)']
                r = metrics['metrics/recall(B)']
                if p + r > 0:
                    metrics['metrics/f1_score'] = 2 * (p * r) / (p + r)
                    
            # Compute improvement over baseline
            if 'metrics/recall(B)' in metrics:
                baseline_recall = 0.568
                metrics['metrics/recall_improvement'] = metrics['metrics/recall(B)'] - baseline_recall
                metrics['metrics/recall_improvement_pct'] = (metrics['metrics/recall(B)'] - baseline_recall) / baseline_recall * 100
            
            if 'metrics/mAP50(B)' in metrics:
                baseline_map50 = 0.636
                metrics['metrics/map50_improvement'] = metrics['metrics/mAP50(B)'] - baseline_map50
                metrics['metrics/map50_improvement_pct'] = (metrics['metrics/mAP50(B)'] - baseline_map50) / baseline_map50 * 100
            
            # Log to wandb
            wandb_run.log(metrics, step=int(row.get('epoch', idx)))
    
    # Log training curves as images
    print("📊 Logging training plots to W&B...")
    for plot_file in results_dir.glob('*.png'):
        try:
            wandb_run.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
        except Exception as e:
            print(f"⚠ Warning: Could not log {plot_file.name} to W&B: {e}")
    
    # Log confusion matrix if available
    confusion_matrix_path = results_dir / 'confusion_matrix.png'
    if confusion_matrix_path.exists():
        wandb_run.log({"confusion_matrix": wandb.Image(str(confusion_matrix_path))})
    
    # Log PR curve if available
    pr_curve_path = results_dir / 'PR_curve.png'
    if pr_curve_path.exists():
        wandb_run.log({"pr_curve": wandb.Image(str(pr_curve_path))})
    
    # Log F1 curve if available
    f1_curve_path = results_dir / 'F1_curve.png'
    if f1_curve_path.exists():
        wandb_run.log({"f1_curve": wandb.Image(str(f1_curve_path))})
    
    # Log best model weights as artifact
    best_weights = results_dir / 'weights' / 'best.pt'
    if best_weights.exists():
        try:
            # Get model size
            model_size_mb = best_weights.stat().st_size / (1024 * 1024)
            
            artifact = wandb.Artifact(
                name=f"model-{wandb_run.name}",
                type="model",
                description=f"Best YOLOv11{args.model} weights for underwater fish detection",
                metadata={
                    "model_size": args.model,
                    "size_mb": round(model_size_mb, 2),
                    "framework": "ultralytics",
                    "task": "object_detection",
                }
            )
            artifact.add_file(str(best_weights), name="best.pt")
            
            # Also add last weights
            last_weights = results_dir / 'weights' / 'last.pt'
            if last_weights.exists():
                artifact.add_file(str(last_weights), name="last.pt")
            
            wandb_run.log_artifact(artifact)
            print(f"✓ Logged model weights to W&B as artifact (size: {model_size_mb:.2f} MB)")
        except Exception as e:
            print(f"⚠ Warning: Could not log weights to W&B: {e}")
    
    # Log final metrics summary
    summary_csv = results_dir / 'metrics_summary.csv'
    if summary_csv.exists():
        try:
            summary_df = pd.read_csv(summary_csv)
            if len(summary_df) > 0:
                final_metrics = summary_df.iloc[-1].to_dict()
                
                # Add final metrics to summary
                summary_dict = {}
                for k, v in final_metrics.items():
                    if k != 'epoch' and pd.notna(v):
                        summary_dict[f"final/{k}"] = float(v)
                
                # Add improvement metrics
                if 'recall' in final_metrics:
                    baseline_recall = 0.568
                    summary_dict['final/recall_improvement'] = float(final_metrics['recall']) - baseline_recall
                    summary_dict['final/recall_improvement_pct'] = (float(final_metrics['recall']) - baseline_recall) / baseline_recall * 100
                    summary_dict['final/recall_target_met'] = float(final_metrics['recall']) >= 0.60
                
                if 'mAP50' in final_metrics:
                    baseline_map50 = 0.636
                    summary_dict['final/map50_improvement'] = float(final_metrics['mAP50']) - baseline_map50
                    summary_dict['final/map50_improvement_pct'] = (float(final_metrics['mAP50']) - baseline_map50) / baseline_map50 * 100
                    summary_dict['final/map50_target_met'] = float(final_metrics['mAP50']) >= 0.65
                
                wandb_run.summary.update(summary_dict)
                print(f"✓ Logged final metrics summary to W&B")
        except Exception as e:
            print(f"⚠ Warning: Could not log summary to W&B: {e}")
    
    # Log training configuration as a table
    try:
        config_data = []
        for key, value in wandb_run.config.items():
            config_data.append([key, str(value)])
        
        config_table = wandb.Table(columns=["Parameter", "Value"], data=config_data)
        wandb_run.log({"configuration": config_table})
    except Exception as e:
        print(f"⚠ Warning: Could not log config table: {e}")


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
    
    # Initialize W&B before training
    wandb_run = setup_wandb(args, train_config)
    
    # Ultralytics has native W&B integration - it will detect the active run
    # and log metrics automatically during training
    if wandb_run is not None:
        os.environ['WANDB_MODE'] = 'online'
        print("✓ Ultralytics W&B integration enabled for real-time logging")
        print("  Metrics will be logged automatically after each epoch")
        print("  📈 View live graphs at:", wandb_run.url)
    
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
    
    # Log additional details to W&B
    if wandb_run is not None:
        print("\n📊 Logging final results to Weights & Biases...")
        log_to_wandb(results_dir, wandb_run, args)
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
        
        # Add derived metrics
        if 'precision' in summary_df.columns and 'recall' in summary_df.columns:
            summary_df['f1_score'] = 2 * (summary_df['precision'] * summary_df['recall']) / (summary_df['precision'] + summary_df['recall'])
        
        # Add improvement metrics
        if 'recall' in summary_df.columns:
            baseline_recall = 0.568
            summary_df['recall_improvement'] = summary_df['recall'] - baseline_recall
        
        if 'mAP50' in summary_df.columns:
            baseline_map50 = 0.636
            summary_df['map50_improvement'] = summary_df['mAP50'] - baseline_map50
        
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
                    value = final[col]
                    if pd.notna(value):
                        print(f"  {col:25s}: {value:.4f}")
            print("-" * 50)
            
            # Print improvement summary
            if 'recall_improvement' in summary_df.columns:
                recall_imp = final['recall_improvement']
                print(f"\n🎯 Performance vs Baseline:")
                print(f"  Recall improvement:     {recall_imp:+.4f} ({recall_imp/0.568*100:+.1f}%)")
                if 'map50_improvement' in summary_df.columns:
                    map_imp = final['map50_improvement']
                    print(f"  mAP@50 improvement:     {map_imp:+.4f} ({map_imp/0.636*100:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 for underwater fish detection with enhanced W&B tracking"
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