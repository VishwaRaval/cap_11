#!/usr/bin/env python3
"""
Training Stability Diagnostic Tool

Analyzes why training is jumpy and suggests fixes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def analyze_training_stability(results_csv_path):
    """
    Analyze training stability from results.csv
    """
    results_csv = Path(results_csv_path)
    if not results_csv.exists():
        print(f"❌ Error: {results_csv} not found")
        return
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    print("\n" + "=" * 80)
    print("TRAINING STABILITY ANALYSIS")
    print("=" * 80)
    
    # Key metrics to analyze
    metrics = {
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall',
        'metrics/mAP50(B)': 'mAP50',
        'val/box_loss': 'Val Box Loss',
        'val/cls_loss': 'Val Cls Loss',
    }
    
    print("\n📊 STABILITY METRICS (Lower = More Stable)\n")
    print(f"{'Metric':<20} {'Std Dev':<12} {'CV (%)':<12} {'Stability':<15}")
    print("-" * 65)
    
    instability_scores = {}
    
    for col, name in metrics.items():
        if col not in df.columns:
            continue
        
        values = df[col].dropna()
        if len(values) < 2:
            continue
        
        # Calculate stability metrics
        mean_val = values.mean()
        std_val = values.std()
        cv = (std_val / mean_val * 100) if mean_val != 0 else float('inf')
        
        # Calculate consecutive differences (jumpiness)
        diffs = values.diff().abs()
        mean_diff = diffs.mean()
        
        # Stability score (lower is better)
        # Penalize: high CV, high consecutive differences
        stability_score = cv * 0.7 + (mean_diff / mean_val * 100) * 0.3 if mean_val != 0 else float('inf')
        instability_scores[name] = stability_score
        
        # Classify stability
        if cv < 5:
            stability = "🟢 Excellent"
        elif cv < 10:
            stability = "🟡 Good"
        elif cv < 20:
            stability = "🟠 Moderate"
        else:
            stability = "🔴 Poor"
        
        print(f"{name:<20} {std_val:>11.4f} {cv:>11.2f} {stability:<15}")
    
    # Overall stability assessment
    print("\n" + "-" * 80)
    avg_instability = np.mean(list(instability_scores.values()))
    
    print(f"\n🎯 OVERALL STABILITY SCORE: {avg_instability:.2f}")
    if avg_instability < 10:
        overall = "🟢 EXCELLENT - Training is very stable"
    elif avg_instability < 20:
        overall = "🟡 GOOD - Training is reasonably stable"
    elif avg_instability < 40:
        overall = "🟠 MODERATE - Training shows some instability"
    else:
        overall = "🔴 POOR - Training is very unstable (JUMPY)"
    
    print(f"   Assessment: {overall}")
    
    # Analyze overfitting
    print("\n📉 OVERFITTING ANALYSIS\n")
    
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        train_loss = df['train/box_loss'].dropna()
        val_loss = df['val/box_loss'].dropna()
        
        if len(train_loss) > 10 and len(val_loss) > 10:
            # Check if validation loss is increasing while training loss decreases
            train_trend = np.polyfit(range(len(train_loss)), train_loss, 1)[0]
            val_trend = np.polyfit(range(len(val_loss)), val_loss, 1)[0]
            
            gap = val_loss.iloc[-1] - train_loss.iloc[-1]
            gap_ratio = gap / train_loss.iloc[-1] * 100
            
            print(f"  Train Loss Trend: {'📉 Decreasing' if train_trend < 0 else '📈 Increasing'}")
            print(f"  Val Loss Trend:   {'📉 Decreasing' if val_trend < 0 else '📈 Increasing'}")
            print(f"  Final Gap:        {gap:.4f} ({gap_ratio:.1f}%)")
            
            if val_trend > 0 and train_trend < 0:
                print(f"  Status:           🔴 OVERFITTING DETECTED")
            elif gap_ratio > 50:
                print(f"  Status:           🟠 High train-val gap (possible overfitting)")
            else:
                print(f"  Status:           🟢 No significant overfitting")
    
    # Convergence analysis
    print("\n📈 CONVERGENCE ANALYSIS\n")
    
    if 'metrics/mAP50(B)' in df.columns:
        mAP50 = df['metrics/mAP50(B)'].dropna()
        
        if len(mAP50) > 20:
            # Check last 20 epochs
            last_20 = mAP50.iloc[-20:]
            improvement = last_20.iloc[-1] - last_20.iloc[0]
            
            print(f"  Last 20 epochs improvement: {improvement*100:+.2f}%")
            
            if abs(improvement) < 0.01:
                print(f"  Status: 🟡 PLATEAU - Model has converged")
            elif improvement > 0.02:
                print(f"  Status: 🟢 IMPROVING - Still learning")
            elif improvement < -0.02:
                print(f"  Status: 🔴 DEGRADING - Model is getting worse")
            else:
                print(f"  Status: 🟡 SLOW PROGRESS - Near convergence")
    
    # Root cause analysis
    print("\n" + "=" * 80)
    print("🔍 ROOT CAUSE ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    issues = []
    recommendations = []
    
    # Check for high instability
    if avg_instability > 40:
        issues.append("❌ Very high training instability (jumpy curves)")
        recommendations.extend([
            "✓ Increase batch size to 64 or higher",
            "✓ Reduce learning rate by 5-10x",
            "✓ Use AdamW optimizer instead of SGD",
            "✓ Increase warmup epochs to 20+",
        ])
    elif avg_instability > 20:
        issues.append("⚠️  Moderate training instability")
        recommendations.extend([
            "✓ Increase batch size to 48-64",
            "✓ Reduce learning rate by 2-3x",
            "✓ Consider AdamW optimizer",
        ])
    
    # Check for overfitting
    if 'val/box_loss' in df.columns:
        val_loss = df['val/box_loss'].dropna()
        if len(val_loss) > 10:
            last_10 = val_loss.iloc[-10:]
            if last_10.iloc[-1] > last_10.iloc[0]:
                issues.append("❌ Validation loss increasing (overfitting)")
                recommendations.extend([
                    "✓ Increase dropout (e.g., 0.2-0.3)",
                    "✓ Increase weight decay",
                    "✓ Reduce augmentation intensity",
                    "✓ Use earlier checkpoint (before overfitting)",
                ])
    
    # Check for lack of convergence
    if 'metrics/mAP50(B)' in df.columns:
        mAP50 = df['metrics/mAP50(B)'].dropna()
        if len(mAP50) > 20:
            last_20 = mAP50.iloc[-20:]
            if last_20.std() > 0.05:
                issues.append("⚠️  High variance in recent epochs")
                recommendations.extend([
                    "✓ Increase batch size for more stable gradients",
                    "✓ Reduce learning rate",
                ])
    
    if issues:
        print("\n🔴 ISSUES DETECTED:\n")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n🟢 NO MAJOR ISSUES DETECTED")
    
    if recommendations:
        print("\n💡 RECOMMENDED FIXES:\n")
        for rec in recommendations:
            print(f"  {rec}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze training stability")
    parser.add_argument('results_csv', type=str, help='Path to results.csv')
    args = parser.parse_args()
    
    analyze_training_stability(args.results_csv)


if __name__ == '__main__':
    main()
