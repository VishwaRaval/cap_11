#!/usr/bin/env python3
"""
Compare different strategies: single models vs ensemble with different thresholds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from test_ensemble_rigorous import test_ensemble_rigorous
from rigorous_eval import evaluate_single_model
import yaml
import argparse


def compare_all_strategies(models, image_dir, labels_dir):
    """
    Compare:
    1. Each single model
    2. Ensemble with different confidence thresholds
    3. Find optimal strategy
    """
    
    results = []
    
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}\n")
    
    # Test each single model
    print("Testing Single Models...")
    print("-" * 80)
    
    for model_path in models:
        model_name = Path(model_path).parent.parent.name
        print(f"\nEvaluating: {model_name}")
        
        result = evaluate_single_model(
            model_path, image_dir, labels_dir,
            conf_threshold=0.25, iou_threshold=0.5
        )
        
        results.append({
            'strategy': f'Single: {model_name}',
            'type': 'single',
            'model': model_name,
            'precision': result['overall']['precision'],
            'recall': result['overall']['recall'],
            'f1': result['overall']['f1'],
            'tp': result['overall']['tp'],
            'fp': result['overall']['fp'],
            'fn': result['overall']['fn']
        })
    
    # Test ensemble with different confidence thresholds
    print(f"\n{'='*80}")
    print("Testing Ensemble with Different Confidence Thresholds...")
    print("-" * 80)
    
    for conf in [0.25, 0.35, 0.45, 0.55]:
        print(f"\nEnsemble (conf={conf}):")
        
        result = test_ensemble_rigorous(
            model_paths=models,
            image_dir=image_dir,
            labels_dir=labels_dir,
            method='wbf',
            conf_threshold=conf,
            iou_threshold=0.5
        )
        
        if result:
            results.append({
                'strategy': f'Ensemble (conf={conf})',
                'type': 'ensemble',
                'conf': conf,
                'precision': result['overall']['precision'],
                'recall': result['overall']['recall'],
                'f1': result['overall']['f1'],
                'tp': result['overall']['tp'],
                'fp': result['overall']['fp'],
                'fn': result['overall']['fn']
            })
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Strategy':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'Status':>10}")
    print("-" * 100)
    
    for r in results:
        prec = r['precision'] * 100
        rec = r['recall'] * 100
        f1 = r['f1'] * 100
        
        # Determine status
        if rec >= 70 and prec >= 70:
            status = "✅ BEST"
        elif rec >= 70:
            status = "⚠️ HIGH FP"
        elif f1 >= 70:
            status = "✅ GOOD"
        else:
            status = "❌ BELOW"
        
        print(f"{r['strategy']:<30} {prec:>7.1f}% {rec:>7.1f}% {f1:>7.1f}% "
              f"{r['tp']:>6d} {r['fp']:>6d} {r['fn']:>6d} {status:>10}")
    
    # Find best strategies
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    # Best F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"🏆 Best Overall (F1): {best_f1['strategy']}")
    print(f"   Precision: {best_f1['precision']*100:.1f}%")
    print(f"   Recall:    {best_f1['recall']*100:.1f}%")
    print(f"   F1:        {best_f1['f1']*100:.1f}%")
    print(f"   FP Rate:   {best_f1['fp']/(best_f1['tp']+best_f1['fp'])*100:.1f}%")
    
    # Best precision (lowest FP)
    best_prec = max(results, key=lambda x: x['precision'])
    print(f"\n🎯 Best Precision (Lowest FP): {best_prec['strategy']}")
    print(f"   Precision: {best_prec['precision']*100:.1f}%")
    print(f"   Recall:    {best_prec['recall']*100:.1f}%")
    print(f"   F1:        {best_prec['f1']*100:.1f}%")
    print(f"   FP Rate:   {best_prec['fp']/(best_prec['tp']+best_prec['fp'])*100:.1f}%")
    
    # Best recall (if needed)
    best_rec = max(results, key=lambda x: x['recall'])
    print(f"\n🔍 Best Recall (Fewest Misses): {best_rec['strategy']}")
    print(f"   Precision: {best_rec['precision']*100:.1f}%")
    print(f"   Recall:    {best_rec['recall']*100:.1f}%")
    print(f"   F1:        {best_rec['f1']*100:.1f}%")
    print(f"   FP Rate:   {best_rec['fp']/(best_rec['tp']+best_rec['fp'])*100:.1f}%")
    
    # Balanced recommendation
    balanced = [r for r in results if r['recall'] >= 0.70 and r['precision'] >= 0.65]
    if balanced:
        best_balanced = max(balanced, key=lambda x: x['f1'])
        print(f"\n⚖️  Best Balanced (Rec≥70%, Prec≥65%): {best_balanced['strategy']}")
        print(f"   Precision: {best_balanced['precision']*100:.1f}%")
        print(f"   Recall:    {best_balanced['recall']*100:.1f}%")
        print(f"   F1:        {best_balanced['f1']*100:.1f}%")
        print(f"   FP Rate:   {best_balanced['fp']/(best_balanced['tp']+best_balanced['fp'])*100:.1f}%")
    
    # Save results
    import json
    with open('strategy_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nFull results saved to: strategy_comparison.json")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data')
    data_group.add_argument('--images')
    
    parser.add_argument('--labels')
    
    args = parser.parse_args()
    
    # Parse paths
    if args.data:
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        
        yaml_dir = Path(args.data).parent
        
        if 'test' in data_config:
            dataset_path = data_config['test']
        elif 'val' in data_config:
            dataset_path = data_config['val']
        else:
            raise ValueError("data.yaml must contain 'test' or 'val'")
        
        if not Path(dataset_path).is_absolute():
            dataset_path = yaml_dir / dataset_path
        
        if (Path(dataset_path) / 'images').exists():
            image_dir = str(Path(dataset_path) / 'images')
            label_dir = str(Path(dataset_path) / 'labels')
        else:
            image_dir = str(dataset_path)
            label_dir = str(Path(dataset_path).parent / 'labels')
    else:
        if not args.labels:
            raise ValueError("--labels required with --images")
        image_dir = args.images
        label_dir = args.labels
    
    compare_all_strategies(args.models, image_dir, label_dir)
