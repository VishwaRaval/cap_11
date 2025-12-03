#!/usr/bin/env python3
"""
Final Model Selection - Which Model Should You Deploy?
Tests top 3 models on TEST SET to find the best one for production
"""

from ultralytics import YOLO
from pathlib import Path
import numpy as np

print('\n' + '='*80)
print('FINAL MODEL SELECTION FOR DEPLOYMENT')
print('='*80)
print('\n🎯 Goal: Find the SINGLE BEST model for accurate bounding boxes + labels\n')

# Top 3 models from comprehensive ranking
models_to_test = [
    {
        'path': 'runs/detect/fish_s_multiscale_heavy_aug_v1/weights/best.pt',
        'name': 'multiscale_heavy_aug',
        'validation_acc': 68.46
    },
    {
        'path': 'runs/detect/extreme_stable_v2_native/weights/best.pt',
        'name': 'extreme_stable_v2',
        'validation_acc': 67.85
    },
    {
        'path': 'runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt',
        'name': 'recall_optimized',
        'validation_acc': 67.24
    },
]

data_yaml = 'dataset_root/data.yaml'

print('Testing models on TEST SET (not validation):')
print('-'*80)

results = []

for i, model_info in enumerate(models_to_test, 1):
    print(f'\n[{i}/3] {model_info["name"]}')
    print(f'      Validation accuracy: {model_info["validation_acc"]:.2f}%')
    print('-'*80)
    
    if not Path(model_info['path']).exists():
        print(f'      ❌ Model not found')
        continue
    
    # Load model
    model = YOLO(model_info['path'])
    
    # Get model size
    model_size_mb = Path(model_info['path']).stat().st_size / (1024 * 1024)
    
    # Test on TEST SET (not validation!)
    print('      Running inference on test set...', end=' ')
    test_results = model.val(
        data=data_yaml,
        split='test',  # Use test set!
        conf=0.20,
        iou=0.5,
        verbose=False
    )
    
    # Extract metrics
    precision = float(test_results.box.mp)
    recall = float(test_results.box.mr)
    mAP50 = float(test_results.box.map50)
    mAP50_95 = float(test_results.box.map)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    avg_acc = (precision + recall) / 2
    
    print('✓')
    
    # Per-class metrics (if available)
    per_class_metrics = {}
    if hasattr(test_results.box, 'p') and hasattr(test_results.box, 'r'):
        class_names = ['Grunt Fish', 'Parrot Fish', 'Surgeon Fish']
        try:
            per_class_p = test_results.box.p
            per_class_r = test_results.box.r
            if len(per_class_p) == 3 and len(per_class_r) == 3:
                per_class_metrics = {
                    name: {'precision': float(p), 'recall': float(r)}
                    for name, p, r in zip(class_names, per_class_p, per_class_r)
                }
        except:
            pass
    
    results.append({
        'name': model_info['name'],
        'path': model_info['path'],
        'size_mb': model_size_mb,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'avg_acc': avg_acc,
        'per_class': per_class_metrics
    })
    
    print(f'\n      Results:')
    print(f'        Precision:    {precision*100:6.2f}%')
    print(f'        Recall:       {recall*100:6.2f}%')
    print(f'        F1 Score:     {f1*100:6.2f}%')
    print(f'        mAP@50:       {mAP50*100:6.2f}%')
    print(f'        mAP@50-95:    {mAP50_95*100:6.2f}%')
    print(f'        Avg Accuracy: {avg_acc*100:6.2f}%')
    print(f'        Model Size:   {model_size_mb:6.1f} MB')
    
    if per_class_metrics:
        print(f'\n      Per-Class Performance:')
        for cls_name, metrics in per_class_metrics.items():
            print(f'        {cls_name:15s}: P={metrics["precision"]*100:5.1f}% R={metrics["recall"]*100:5.1f}%')

# Rank results
results.sort(key=lambda x: x['avg_acc'], reverse=True)

# Final recommendation
print('\n' + '='*80)
print('📊 FINAL COMPARISON (TEST SET)')
print('='*80)

print(f"\n{'Rank':<6} {'Model':<30} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'Avg Acc':<12}")
print('-'*80)

for i, r in enumerate(results, 1):
    print(f"{i:<6} {r['name']:<30} {r['precision']*100:<11.2f}% {r['recall']*100:<11.2f}% "
          f"{r['mAP50']*100:<11.2f}% {r['avg_acc']*100:<11.2f}%")

print('-'*80)

# Best model
best = results[0]

print('\n' + '='*80)
print('🏆 RECOMMENDED MODEL FOR DEPLOYMENT')
print('='*80)

print(f'\nModel: {best["name"]}')
print(f'Path:  {best["path"]}')
print()
print('Performance on TEST SET:')
print(f'  Precision:    {best["precision"]*100:.2f}%')
print(f'  Recall:       {best["recall"]*100:.2f}%')
print(f'  F1 Score:     {best["f1"]*100:.2f}%')
print(f'  mAP@50:       {best["mAP50"]*100:.2f}%')
print(f'  mAP@50-95:    {best["mAP50_95"]*100:.2f}%')
print(f'  Avg Accuracy: {best["avg_acc"]*100:.2f}%')
print()
print(f'Model Size: {best["size_mb"]:.1f} MB')
print(f'Size Check: {"✅ Under 70MB limit" if best["size_mb"] < 70 else "❌ Exceeds 70MB limit"}')

if best['per_class']:
    print('\nPer-Class Performance:')
    for cls_name, metrics in best['per_class'].items():
        p = metrics['precision']
        r = metrics['recall']
        f1_cls = 2 * p * r / (p + r + 1e-8)
        print(f'  {cls_name:15s}: Precision={p*100:5.1f}% Recall={r*100:5.1f}% F1={f1_cls*100:5.1f}%')

# Quality assessment
print('\n' + '='*80)
print('📈 QUALITY ASSESSMENT')
print('='*80)

if best['avg_acc'] >= 0.70:
    quality = '✅ EXCELLENT - Meets 70% target!'
    recommendation = 'Deploy this model immediately.'
elif best['avg_acc'] >= 0.68:
    quality = '✓ VERY GOOD - Close to target'
    recommendation = 'This model is production-ready. Gap to 70% is within acceptable range.'
elif best['avg_acc'] >= 0.65:
    quality = '⚠️  GOOD - Below target but usable'
    recommendation = 'Acceptable for deployment. Monitor performance and consider improvements.'
else:
    quality = '❌ NEEDS IMPROVEMENT'
    recommendation = 'Not recommended for production. Consider more training or different approach.'

print(f'\nQuality: {quality}')
print(f'Test Accuracy: {best["avg_acc"]*100:.2f}%')
print(f'Target: 70.00%')

if best['avg_acc'] < 0.70:
    print(f'Gap: {(0.70 - best["avg_acc"])*100:.2f}%')

print(f'\nRecommendation: {recommendation}')

# Precision vs Recall analysis
print('\n' + '='*80)
print('⚖️  PRECISION vs RECALL TRADE-OFF')
print('='*80)

prec_rec_ratio = best['precision'] / (best['recall'] + 1e-8)

print(f'\nPrecision: {best["precision"]*100:.2f}%')
print(f'Recall:    {best["recall"]*100:.2f}%')
print(f'Ratio:     {prec_rec_ratio:.2f}')

if prec_rec_ratio > 1.2:
    print('\n✓ HIGH PRECISION model')
    print('  - Few false positives (good!)')
    print('  - May miss some fish (lower recall)')
    print('  - Best for: Applications where false alarms are costly')
elif prec_rec_ratio < 0.8:
    print('\n⚠️  HIGH RECALL model')
    print('  - Finds most fish (good!)')
    print('  - More false positives (precision lower)')
    print('  - Best for: Applications where missing fish is costly')
else:
    print('\n✅ BALANCED model')
    print('  - Good balance between precision and recall')
    print('  - Best for: General purpose fish detection')

# Deployment instructions
print('\n' + '='*80)
print('🚀 DEPLOYMENT INSTRUCTIONS')
print('='*80)

print(f'\n1. Use this model file:')
print(f'   {best["path"]}')

print(f'\n2. Recommended inference settings:')
print(f'   - Confidence threshold: 0.20')
print(f'   - IoU threshold: 0.50')
print(f'   - Input size: 768×432 (native) or 768×768 (square)')

print(f'\n3. Python code for inference:')
print(f'```python')
print(f'from ultralytics import YOLO')
print(f'')
print(f'# Load model')
print(f'model = YOLO("{best["path"]}")')
print(f'')
print(f'# Run inference')
print(f'results = model.predict(')
print(f'    source="your_image.jpg",')
print(f'    conf=0.20,')
print(f'    iou=0.50,')
print(f'    save=True  # Save annotated images')
print(f')')
print(f'```')

print(f'\n4. Expected performance:')
print(f'   - Approximately {best["avg_acc"]*100:.0f}% accurate classification')
print(f'   - mAP@50 of {best["mAP50"]*100:.0f}%')
print(f'   - Model size: {best["size_mb"]:.1f} MB')

print('\n' + '='*80)
print('✓ EVALUATION COMPLETE')
print('='*80)

# Save recommendation
with open('deployment_recommendation.txt', 'w') as f:
    f.write('FINAL MODEL RECOMMENDATION FOR DEPLOYMENT\n')
    f.write('='*80 + '\n\n')
    f.write(f'Selected Model: {best["name"]}\n')
    f.write(f'Path: {best["path"]}\n')
    f.write(f'Test Accuracy: {best["avg_acc"]*100:.2f}%\n')
    f.write(f'Precision: {best["precision"]*100:.2f}%\n')
    f.write(f'Recall: {best["recall"]*100:.2f}%\n')
    f.write(f'mAP@50: {best["mAP50"]*100:.2f}%\n')
    f.write(f'Model Size: {best["size_mb"]:.1f} MB\n')
    f.write(f'\nRecommendation: {recommendation}\n')

print('\n✓ Deployment recommendation saved to: deployment_recommendation.txt')
print('='*80)
