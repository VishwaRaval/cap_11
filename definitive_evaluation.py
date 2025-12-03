#!/usr/bin/env python3
"""
DEFINITIVE Model Evaluation - Consistent Settings
Uses EXACT same settings across all evaluations for fair comparison
"""

from ultralytics import YOLO
from pathlib import Path

print('\n' + '='*80)
print('DEFINITIVE MODEL EVALUATION - CONSISTENT SETTINGS')
print('='*80)
print('\n🎯 Goal: Get TRUE performance with consistent evaluation\n')

# Top 3 models
models = [
    {
        'path': 'runs/detect/fish_s_multiscale_heavy_aug_v1/weights/best.pt',
        'name': 'multiscale_heavy_aug'
    },
    {
        'path': 'runs/detect/extreme_stable_v2_native/weights/best.pt',
        'name': 'extreme_stable_v2'
    },
    {
        'path': 'runs/detect/extreme_stable_v1/weights/best.pt',
        'name': 'extreme_stable_v1'
    },
]

data_yaml = 'dataset_root/data.yaml'

print('SETTINGS (matching compare_models_py.py):')
print('-'*80)
print('  Image size: 768')
print('  Batch size: 16')
print('  Split: test')
print('  Confidence: default (model\'s trained threshold)')
print('  IoU: default (0.5)')
print('-'*80)

results = []

for model_info in models:
    print(f'\n📊 {model_info["name"]}')
    print('-'*80)
    
    if not Path(model_info['path']).exists():
        print('  ❌ Model not found')
        continue
    
    model = YOLO(model_info['path'])
    model_size_mb = Path(model_info['path']).stat().st_size / (1024 * 1024)
    
    # Validate on TEST set with DEFAULT settings (like compare_models.py)
    print('  Validating on test set...', end=' ')
    test_results = model.val(
        data=data_yaml,
        split='test',
        imgsz=768,
        batch=16,
        plots=False,
        save_json=False,
        verbose=False
    )
    print('✓')
    
    # Extract metrics
    metrics_dict = test_results.results_dict
    precision = metrics_dict.get('metrics/precision(B)', 0)
    recall = metrics_dict.get('metrics/recall(B)', 0)
    mAP50 = metrics_dict.get('metrics/mAP50(B)', 0)
    mAP50_95 = metrics_dict.get('metrics/mAP50-95(B)', 0)
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_acc = (precision + recall) / 2
    
    results.append({
        'name': model_info['name'],
        'path': model_info['path'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'avg_acc': avg_acc,
        'size_mb': model_size_mb
    })
    
    print(f'\n  Results:')
    print(f'    Precision:    {precision*100:6.2f}%')
    print(f'    Recall:       {recall*100:6.2f}%')
    print(f'    F1 Score:     {f1*100:6.2f}%')
    print(f'    mAP50:        {mAP50*100:6.2f}%')
    print(f'    mAP50-95:     {mAP50_95*100:6.2f}%')
    print(f'    Avg Accuracy: {avg_acc*100:6.2f}%')
    print(f'    Size:         {model_size_mb:6.1f} MB')

# Sort by avg_acc
results.sort(key=lambda x: x['avg_acc'], reverse=True)

# Final table
print('\n' + '='*80)
print('📊 FINAL RANKING (TEST SET)')
print('='*80)

print(f"\n{'Rank':<6} {'Model':<30} {'Precision':<11} {'Recall':<11} {'mAP50':<11} {'Avg Acc':<11}")
print('-'*80)

for i, r in enumerate(results, 1):
    target = '✅' if r['avg_acc'] >= 0.70 else '❌'
    print(f"{i:<6} {r['name']:<30} {r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% "
          f"{r['mAP50']*100:>9.2f}% {r['avg_acc']*100:>9.2f}% {target}")

print('-'*80)

# Best model
best = results[0]

print('\n' + '='*80)
print('🏆 BEST MODEL FOR DEPLOYMENT')
print('='*80)

print(f'\nModel: {best["name"]}')
print(f'Path:  {best["path"]}')
print(f'\nTest Set Performance:')
print(f'  Precision:    {best["precision"]*100:.2f}%')
print(f'  Recall:       {best["recall"]*100:.2f}%')
print(f'  F1 Score:     {best["f1"]*100:.2f}%')
print(f'  mAP@50:       {best["mAP50"]*100:.2f}%')
print(f'  Avg Accuracy: {best["avg_acc"]*100:.2f}%')
print(f'  Model Size:   {best["size_mb"]:.1f} MB')

print(f'\n🎯 Target: 70.00%')
print(f'   Actual: {best["avg_acc"]*100:.2f}%')

if best['avg_acc'] >= 0.70:
    print(f'   Status: ✅ TARGET MET!')
else:
    print(f'   Gap:    {(0.70 - best["avg_acc"])*100:.2f}%')
    print(f'   Status: ❌ Below target')

# Reality check
print('\n' + '='*80)
print('🔍 REALITY CHECK')
print('='*80)

print('\nDocument 26 shows these models at:')
print('  extreme_stable_v1:        65.28% (on some split)')
print('  extreme_stable_v2_native: 64.27% (on some split)')
print('  multiscale_heavy_aug:     57.75% (on some split)')

print(f'\nThis script shows (TEST SET):')
for r in results:
    print(f'  {r["name"]:<30}: {r["avg_acc"]*100:.2f}%')

print('\nDifferences likely due to:')
print('  1. Different splits (valid vs test)')
print('  2. Different confidence thresholds')
print('  3. Different evaluation settings')

print('\n💡 RECOMMENDATION:')
print('  Use THIS script\'s results as the definitive truth')
print('  These are test set results with consistent settings')

print('\n' + '='*80)
print('DEPLOYMENT DECISION')
print('='*80)

if best['avg_acc'] >= 0.70:
    print('\n✅ Deploy this model immediately!')
    print(f'   Model: {best["name"]}')
    print(f'   Test Accuracy: {best["avg_acc"]*100:.2f}%')
elif best['avg_acc'] >= 0.65:
    print('\n⚠️  Model is GOOD but below 70% target')
    print(f'   Model: {best["name"]}')
    print(f'   Test Accuracy: {best["avg_acc"]*100:.2f}%')
    print(f'   Gap to target: {(0.70 - best["avg_acc"])*100:.2f}%')
    print('\n   Options:')
    print('   1. Accept this performance (65%+ is decent for this task)')
    print('   2. Use ensemble of top 3 models')
    print('   3. Accept that 70% may not be achievable with current setup')
else:
    print('\n❌ Performance below expectations')
    print(f'   Best: {best["avg_acc"]*100:.2f}%')
    print('   Need different approach')

print('\n' + '='*80)

# Save results
with open('definitive_results.txt', 'w') as f:
    f.write('DEFINITIVE MODEL EVALUATION RESULTS\n')
    f.write('='*80 + '\n\n')
    f.write('Test Set Performance:\n\n')
    for i, r in enumerate(results, 1):
        f.write(f"{i}. {r['name']}\n")
        f.write(f"   Precision: {r['precision']*100:.2f}%\n")
        f.write(f"   Recall: {r['recall']*100:.2f}%\n")
        f.write(f"   Avg Acc: {r['avg_acc']*100:.2f}%\n")
        f.write(f"   mAP50: {r['mAP50']*100:.2f}%\n\n")
    
    f.write(f"\nRECOMMENDATION: Deploy {best['name']}\n")
    f.write(f"Test Accuracy: {best['avg_acc']*100:.2f}%\n")

print('✓ Results saved to: definitive_results.txt\n')
print('='*80)
