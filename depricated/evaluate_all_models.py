#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Comparison
Evaluates all trained models on test set and ranks them
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict
import yaml


def find_all_best_weights(runs_dir='runs/detect'):
    """Find all best.pt weights in the runs directory."""
    runs_dir = Path(runs_dir)
    weights = []
    
    for exp_dir in runs_dir.glob('fish_*'):
        best_pt = exp_dir / 'weights' / 'best.pt'
        if best_pt.exists():
            weights.append({
                'name': exp_dir.name,
                'path': str(best_pt),
                'exp_dir': str(exp_dir)
            })
    
    return sorted(weights, key=lambda x: x['name'])


def evaluate_model(model_path, data_yaml, split='test'):
    """Evaluate a single model."""
    model = YOLO(model_path)
    
    # Run validation on test set
    results = model.val(
        data=data_yaml,
        split=split,
        batch=16,
        imgsz=768,
        verbose=False,
        plots=False,
    )
    
    # Extract metrics
    metrics = {
        'precision': results.box.mp,  # mean precision
        'recall': results.box.mr,     # mean recall
        'mAP50': results.box.map50,
        'mAP50_95': results.box.map,
        'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0,
        'avg_accuracy': (results.box.mp + results.box.mr) / 2,
    }
    
    # Per-class metrics if available
    if hasattr(results.box, 'p') and results.box.p is not None:
        metrics['per_class_precision'] = results.box.p.tolist()
    if hasattr(results.box, 'r') and results.box.r is not None:
        metrics['per_class_recall'] = results.box.r.tolist()
    if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
        metrics['per_class_ap50'] = results.box.ap50.tolist()
    
    return metrics


def load_training_metadata(exp_dir):
    """Load training configuration and final epoch metrics."""
    exp_dir = Path(exp_dir)
    
    metadata = {}
    
    # Try to load args.yaml
    args_yaml = exp_dir / 'args.yaml'
    if args_yaml.exists():
        with open(args_yaml, 'r') as f:
            args = yaml.safe_load(f)
            metadata['model'] = args.get('model', 'unknown')
            metadata['epochs_trained'] = args.get('epochs', 0)
            metadata['batch_size'] = args.get('batch', 0)
    
    # Try to load results.csv for final epoch metrics
    results_csv = exp_dir / 'results.csv'
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        if len(df) > 0:
            final = df.iloc[-1]
            metadata['final_train_loss'] = final.get('train/box_loss', 0) + final.get('train/cls_loss', 0)
            metadata['final_val_loss'] = final.get('val/box_loss', 0) + final.get('val/cls_loss', 0)
    
    return metadata


def rank_models(results_df):
    """Rank models by multiple criteria."""
    rankings = {}
    
    # Primary: Average accuracy (closest to 70% target)
    results_df['target_gap'] = abs(results_df['avg_accuracy'] - 0.70)
    rankings['by_target'] = results_df.nsmallest(10, 'target_gap')[['name', 'avg_accuracy', 'target_gap']]
    
    # Secondary: Highest F1 score
    rankings['by_f1'] = results_df.nlargest(10, 'f1')[['name', 'f1', 'precision', 'recall']]
    
    # Tertiary: Best recall (important for detection)
    rankings['by_recall'] = results_df.nlargest(10, 'recall')[['name', 'recall', 'precision', 'mAP50']]
    
    # Quaternary: Best mAP50
    rankings['by_map50'] = results_df.nlargest(10, 'mAP50')[['name', 'mAP50', 'avg_accuracy']]
    
    return rankings


def print_comparison_table(results_df):
    """Print comprehensive comparison table."""
    print("\n" + "="*100)
    print("COMPREHENSIVE MODEL COMPARISON - TEST SET RESULTS")
    print("="*100)
    
    # Sort by average accuracy (descending)
    results_sorted = results_df.sort_values('avg_accuracy', ascending=False)
    
    print(f"\n{'Model':<35} {'Prec':<8} {'Rec':<8} {'F1':<8} {'mAP50':<8} {'Avg_Acc':<8} {'Target':<8}")
    print("-"*100)
    
    for _, row in results_sorted.iterrows():
        name = row['name'][:33]
        prec = f"{row['precision']*100:.2f}%"
        rec = f"{row['recall']*100:.2f}%"
        f1 = f"{row['f1']*100:.2f}%"
        map50 = f"{row['mAP50']*100:.2f}%"
        avg_acc = f"{row['avg_accuracy']*100:.2f}%"
        target_met = '✅' if row['avg_accuracy'] >= 0.70 else '❌'
        
        print(f"{name:<35} {prec:<8} {rec:<8} {f1:<8} {map50:<8} {avg_acc:<8} {target_met:<8}")
    
    print("="*100)


def print_top_performers(rankings):
    """Print top performers by different criteria."""
    print("\n" + "="*100)
    print("TOP PERFORMERS BY DIFFERENT CRITERIA")
    print("="*100)
    
    # Best for target
    print("\n🎯 CLOSEST TO 70% TARGET:")
    print("-"*100)
    for idx, row in rankings['by_target'].head(5).iterrows():
        gap = row['target_gap'] * 100
        acc = row['avg_accuracy'] * 100
        print(f"  {idx+1}. {row['name'][:50]:<50} | Acc: {acc:.2f}% | Gap: {gap:.2f}%")
    
    # Best F1
    print("\n📊 HIGHEST F1 SCORE:")
    print("-"*100)
    for idx, row in rankings['by_f1'].head(5).iterrows():
        print(f"  {idx+1}. {row['name'][:50]:<50} | F1: {row['f1']*100:.2f}% | P: {row['precision']*100:.2f}% | R: {row['recall']*100:.2f}%")
    
    # Best recall
    print("\n🔍 HIGHEST RECALL (Detection Sensitivity):")
    print("-"*100)
    for idx, row in rankings['by_recall'].head(5).iterrows():
        print(f"  {idx+1}. {row['name'][:50]:<50} | Recall: {row['recall']*100:.2f}% | mAP50: {row['mAP50']*100:.2f}%")
    
    print("="*100)


def analyze_per_class_performance(results_df, class_names=['Grunt Fish', 'Parrot Fish', 'Surgeon Fish']):
    """Analyze per-class performance for top models."""
    print("\n" + "="*100)
    print("PER-CLASS PERFORMANCE ANALYSIS - TOP 3 MODELS")
    print("="*100)
    
    # Get top 3 by avg accuracy
    top_models = results_df.nlargest(3, 'avg_accuracy')
    
    for idx, row in top_models.iterrows():
        print(f"\n{idx+1}. {row['name']}")
        print("-"*100)
        
        if 'per_class_precision' in row and row['per_class_precision'] is not None:
            prec = row['per_class_precision']
            rec = row['per_class_recall'] if 'per_class_recall' in row else [0, 0, 0]
            ap50 = row['per_class_ap50'] if 'per_class_ap50' in row else [0, 0, 0]
            
            print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'AP50':<12}")
            print("-"*60)
            for i, cls_name in enumerate(class_names):
                p = prec[i] * 100 if i < len(prec) else 0
                r = rec[i] * 100 if i < len(rec) else 0
                a = ap50[i] * 100 if i < len(ap50) else 0
                print(f"{cls_name:<20} {p:>10.2f}%  {r:>10.2f}%  {a:>10.2f}%")
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare all trained models")
    parser.add_argument('--runs-dir', type=str, default='runs/detect',
                       help='Directory containing experiment runs')
    parser.add_argument('--data', type=str, default='dataset_root/data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--split', type=str, default='test',
                       choices=['test', 'val'],
                       help='Split to evaluate on')
    parser.add_argument('--output', type=str, default='model_comparison.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Find all models
    print("🔍 Searching for trained models...")
    models = find_all_best_weights(args.runs_dir)
    
    if not models:
        print(f"❌ No models found in {args.runs_dir}")
        return
    
    print(f"✓ Found {len(models)} models\n")
    
    # Evaluate each model
    results = []
    
    print("=" * 100)
    print("EVALUATING ALL MODELS")
    print("=" * 100 + "\n")
    
    for i, model_info in enumerate(models, 1):
        print(f"[{i}/{len(models)}] Evaluating: {model_info['name']}")
        
        try:
            metrics = evaluate_model(model_info['path'], args.data, args.split)
            metadata = load_training_metadata(model_info['exp_dir'])
            
            result = {
                'name': model_info['name'],
                'path': model_info['path'],
                **metrics,
                **metadata,
            }
            results.append(result)
            
            print(f"  ✓ Precision: {metrics['precision']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | mAP50: {metrics['mAP50']*100:.2f}%")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue
    
    if not results:
        print("\n❌ No models evaluated successfully")
        return
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(args.output, index=False, float_format='%.6f')
    print(f"\n✓ Results saved to: {args.output}")
    
    # Rankings
    rankings = rank_models(results_df)
    
    # Print comprehensive comparison
    print_comparison_table(results_df)
    
    # Print top performers
    print_top_performers(rankings)
    
    # Per-class analysis
    analyze_per_class_performance(results_df)
    
    # Final recommendations
    print("\n" + "="*100)
    print("FINAL RECOMMENDATIONS")
    print("="*100)
    
    best_model = results_df.loc[results_df['avg_accuracy'].idxmax()]
    best_recall = results_df.loc[results_df['recall'].idxmax()]
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    
    print(f"\n🏆 BEST OVERALL (Avg Accuracy): {best_model['name']}")
    print(f"   Accuracy: {best_model['avg_accuracy']*100:.2f}% | Precision: {best_model['precision']*100:.2f}% | Recall: {best_model['recall']*100:.2f}%")
    print(f"   Path: {best_model['path']}")
    
    print(f"\n🔍 BEST FOR DETECTION (Recall): {best_recall['name']}")
    print(f"   Recall: {best_recall['recall']*100:.2f}% | Precision: {best_recall['precision']*100:.2f}%")
    
    print(f"\n⚖️  BEST BALANCED (F1): {best_f1['name']}")
    print(f"   F1: {best_f1['f1']*100:.2f}% | Precision: {best_f1['precision']*100:.2f}% | Recall: {best_f1['recall']*100:.2f}%")
    
    if best_model['avg_accuracy'] >= 0.70:
        print(f"\n🎉 TARGET ACHIEVED! Model exceeds 70% average accuracy!")
    else:
        gap = (0.70 - best_model['avg_accuracy']) * 100
        print(f"\n📈 Gap to 70% target: {gap:.2f}%")
        print("   Consider: Larger model, more data, or ensemble methods")
    
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
