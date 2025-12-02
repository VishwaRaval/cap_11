#!/usr/bin/env python3
"""
Real-time Training Monitor
Watches the latest experiment and reports progress
"""

import time
import pandas as pd
from pathlib import Path
from datetime import datetime


def get_latest_experiment(runs_dir='runs/detect'):
    """Find the most recently modified experiment."""
    runs_dir = Path(runs_dir)
    
    if not runs_dir.exists():
        return None
    
    experiments = [d for d in runs_dir.glob('fish_*') if d.is_dir()]
    
    if not experiments:
        return None
    
    # Sort by modification time
    latest = max(experiments, key=lambda x: x.stat().st_mtime)
    return latest


def read_results(exp_dir):
    """Read current results from CSV."""
    results_csv = exp_dir / 'results.csv'
    
    if not results_csv.exists():
        return None
    
    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        return df
    except:
        return None


def format_metrics(row):
    """Format metrics row for display."""
    epoch = int(row.get('epoch', 0))
    
    # Training losses
    train_box = row.get('train/box_loss', 0)
    train_cls = row.get('train/cls_loss', 0)
    train_dfl = row.get('train/dfl_loss', 0)
    
    # Validation losses
    val_box = row.get('val/box_loss', 0)
    val_cls = row.get('val/cls_loss', 0)
    val_dfl = row.get('val/dfl_loss', 0)
    
    # Metrics
    precision = row.get('metrics/precision(B)', 0) * 100
    recall = row.get('metrics/recall(B)', 0) * 100
    map50 = row.get('metrics/mAP50(B)', 0) * 100
    map50_95 = row.get('metrics/mAP50-95(B)', 0) * 100
    
    avg_acc = (precision + recall) / 2
    gap_to_70 = 70.0 - avg_acc
    
    return {
        'epoch': epoch,
        'train_box': train_box,
        'train_cls': train_cls,
        'val_box': val_box,
        'val_cls': val_cls,
        'precision': precision,
        'recall': recall,
        'map50': map50,
        'avg_acc': avg_acc,
        'gap_to_70': gap_to_70,
    }


def print_header():
    """Print table header."""
    print("\n" + "="*120)
    print(f"{'Epoch':<6} {'Train Loss':<15} {'Val Loss':<15} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'Avg_Acc':<10} {'Gap_70%':<10}")
    print("-"*120)


def print_row(metrics):
    """Print metrics row."""
    epoch = metrics['epoch']
    train_loss = f"{metrics['train_box']:.3f}+{metrics['train_cls']:.3f}"
    val_loss = f"{metrics['val_box']:.3f}+{metrics['val_cls']:.3f}"
    prec = f"{metrics['precision']:.2f}%"
    rec = f"{metrics['recall']:.2f}%"
    map50 = f"{metrics['map50']:.2f}%"
    avg_acc = f"{metrics['avg_acc']:.2f}%"
    
    # Color code gap
    gap = metrics['gap_to_70']
    if gap <= 0:
        gap_str = f"✅ TARGET MET"
    elif gap <= 5:
        gap_str = f"⚡ {gap:.2f}%"
    else:
        gap_str = f"{gap:.2f}%"
    
    print(f"{epoch:<6} {train_loss:<15} {val_loss:<15} {prec:<10} {rec:<10} {map50:<10} {avg_acc:<10} {gap_str:<10}")


def print_summary(df, exp_name):
    """Print summary statistics."""
    if len(df) == 0:
        return
    
    latest = df.iloc[-1]
    metrics = format_metrics(latest)
    
    print("\n" + "="*120)
    print(f"CURRENT EXPERIMENT: {exp_name}")
    print("="*120)
    print(f"Epochs completed: {int(latest.get('epoch', 0))}")
    print(f"Latest metrics:")
    print(f"  Precision:    {metrics['precision']:.2f}%")
    print(f"  Recall:       {metrics['recall']:.2f}%")
    print(f"  mAP50:        {metrics['map50']:.2f}%")
    print(f"  Avg Accuracy: {metrics['avg_acc']:.2f}%")
    print(f"  Gap to 70%:   {metrics['gap_to_70']:.2f}%")
    
    # Best metrics so far
    if 'metrics/precision(B)' in df.columns:
        best_prec = df['metrics/precision(B)'].max() * 100
        best_rec = df['metrics/recall(B)'].max() * 100
        best_map = df['metrics/mAP50(B)'].max() * 100
        
        print(f"\nBest metrics so far:")
        print(f"  Best Precision: {best_prec:.2f}%")
        print(f"  Best Recall:    {best_rec:.2f}%")
        print(f"  Best mAP50:     {best_map:.2f}%")
    
    # Trend analysis (last 5 epochs)
    if len(df) >= 5:
        recent = df.tail(5)
        prec_trend = recent['metrics/precision(B)'].mean() * 100
        rec_trend = recent['metrics/recall(B)'].mean() * 100
        
        print(f"\nRecent trend (last 5 epochs):")
        print(f"  Avg Precision: {prec_trend:.2f}%")
        print(f"  Avg Recall:    {rec_trend:.2f}%")
    
    print("="*120)


def monitor_training(interval=30):
    """Monitor training progress."""
    print("="*120)
    print("REAL-TIME TRAINING MONITOR")
    print("="*120)
    print("Watching: runs/detect/")
    print(f"Update interval: {interval}s")
    print("Press Ctrl+C to stop")
    print("="*120)
    
    last_epoch = -1
    last_exp = None
    
    try:
        while True:
            # Find latest experiment
            exp_dir = get_latest_experiment()
            
            if exp_dir is None:
                print("⏳ Waiting for training to start...")
                time.sleep(interval)
                continue
            
            # Check if experiment changed
            if last_exp is None or exp_dir != last_exp:
                last_exp = exp_dir
                last_epoch = -1
                print(f"\n🚀 Detected new experiment: {exp_dir.name}")
                print_header()
            
            # Read results
            df = read_results(exp_dir)
            
            if df is None or len(df) == 0:
                time.sleep(interval)
                continue
            
            # Check for new epochs
            current_epoch = int(df.iloc[-1].get('epoch', 0))
            
            if current_epoch > last_epoch:
                # Print all new epochs
                new_rows = df[df['epoch'] > last_epoch]
                
                for _, row in new_rows.iterrows():
                    metrics = format_metrics(row)
                    print_row(metrics)
                
                last_epoch = current_epoch
                
                # Print summary every 10 epochs
                if current_epoch % 10 == 0:
                    print_summary(df, exp_dir.name)
                    print_header()
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped")
        
        if last_exp:
            df = read_results(last_exp)
            if df is not None:
                print_summary(df, last_exp.name)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor training progress in real-time")
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    monitor_training(args.interval)
