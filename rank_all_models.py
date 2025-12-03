#!/usr/bin/env python3
"""
Comprehensive Model Ranking
Evaluates ALL trained models and ranks by performance
"""

from ultralytics import YOLO
from pathlib import Path
import glob

print('\n' + '='*80)
print('COMPREHENSIVE MODEL RANKING')
print('='*80)
print('\nScanning for all trained models...\n')

# Find all best.pt files in runs/detect
base_path = Path('runs/detect')
model_paths = list(base_path.glob('*/weights/best.pt'))

if not model_paths:
    print("❌ No models found in runs/detect/*/weights/best.pt")
    exit(1)

print(f'✓ Found {len(model_paths)} trained models\n')

# Validate each model
print('='*80)
print('VALIDATING ALL MODELS')
print('='*80)

results = []
data_yaml = 'dataset_root/data.yaml'

for i, model_path in enumerate(model_paths, 1):
    exp_name = model_path.parent.parent.name
    print(f'\n[{i}/{len(model_paths)}] {exp_name}')
    print('-'*80)
    
    try:
        model = YOLO(str(model_path))
        
        # Validate with standard settings
        val_results = model.val(
            data=data_yaml,
            conf=0.20,
            iou=0.5,
            verbose=False
        )
        
        # Extract metrics
        precision = float(val_results.box.mp)
        recall = float(val_results.box.mr)
        mAP50 = float(val_results.box.map50)
        mAP50_95 = float(val_results.box.map)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        avg_acc = (precision + recall) / 2
        
        # Get model size
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        results.append({
            'name': exp_name,
            'path': str(model_path),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mAP50': mAP50,
            'mAP50_95': mAP50_95,
            'avg_acc': avg_acc,
            'size_mb': model_size_mb
        })
        
        print(f'  Precision: {precision*100:5.2f}%  Recall: {recall*100:5.2f}%  '
              f'Avg Acc: {avg_acc*100:5.2f}%  Size: {model_size_mb:.1f}MB')
        
    except Exception as e:
        print(f'  ❌ Error: {e}')
        continue

# Sort by avg_acc
results.sort(key=lambda x: x['avg_acc'], reverse=True)

# Print comprehensive ranking
print('\n' + '='*80)
print('📊 COMPLETE MODEL RANKING (Sorted by Avg Accuracy)')
print('='*80)
print()

print('┌' + '─'*78 + '┐')
print(f"│ {'#':<3} {'MODEL':<35} {'PREC':>7} {'REC':>7} {'F1':>7} {'AVG':>7} {'SIZE':>6} │")
print('├' + '─'*78 + '┤')

for i, r in enumerate(results, 1):
    target_marker = '✅' if r['avg_acc'] >= 0.70 else '❌'
    print(f"│ {i:<3} {r['name'][:33]:<35} {r['precision']*100:>6.2f}% {r['recall']*100:>6.2f}% "
          f"{r['f1']*100:>6.2f}% {r['avg_acc']*100:>6.2f}% {r['size_mb']:>5.1f}M │")

print('└' + '─'*78 + '┘')

# Top performers
print('\n' + '='*80)
print('🏆 TOP 5 MODELS')
print('='*80)

for i, r in enumerate(results[:5], 1):
    print(f'\n{i}. {r["name"]}')
    print('-'*80)
    print(f'   Precision:   {r["precision"]*100:.2f}%')
    print(f'   Recall:      {r["recall"]*100:.2f}%')
    print(f'   F1 Score:    {r["f1"]*100:.2f}%')
    print(f'   mAP50:       {r["mAP50"]*100:.2f}%')
    print(f'   mAP50-95:    {r["mAP50_95"]*100:.2f}%')
    print(f'   Avg Acc:     {r["avg_acc"]*100:.2f}%')
    print(f'   Size:        {r["size_mb"]:.1f} MB')
    print(f'   Path:        {r["path"]}')

# Best model analysis
best = results[0]
print('\n' + '='*80)
print('🥇 BEST SINGLE MODEL')
print('='*80)
print(f'\nModel: {best["name"]}')
print(f'Performance: {best["avg_acc"]*100:.2f}%')
print(f'Gap to 70%: {(0.70 - best["avg_acc"])*100:.2f}%')

if best['avg_acc'] >= 0.70:
    print('\n✅ TARGET MET!')
    print('   This single model is ready for deployment!')
else:
    print(f'\n❌ Below target by {(0.70 - best["avg_acc"])*100:.2f}%')

# Ensemble potential
print('\n' + '='*80)
print('📊 ENSEMBLE ANALYSIS')
print('='*80)

top_4 = results[:4]
avg_precision = sum(r['precision'] for r in top_4) / len(top_4)
avg_recall = sum(r['recall'] for r in top_4) / len(top_4)
simple_avg = (avg_precision + avg_recall) / 2

print(f'\nTop 4 models simple average: {simple_avg*100:.2f}%')
print(f'Expected true ensemble: {(simple_avg + 0.015)*100:.2f}% (simple avg + 1.5%)')
print(f'Optimistic ensemble: {(simple_avg + 0.025)*100:.2f}% (simple avg + 2.5%)')

# Check if any model is significantly different
precisions = [r['precision'] for r in top_4]
recalls = [r['recall'] for r in top_4]
import numpy as np
prec_std = np.std(precisions)
rec_std = np.std(recalls)
diversity = (prec_std + rec_std) / 2

print(f'\nModel diversity: {diversity:.4f}')
if diversity < 0.02:
    print('  ⚠️  LOW diversity - models are very similar')
    print('      Ensemble gain will be minimal (<1%)')
elif diversity < 0.04:
    print('  ✓ MODERATE diversity - expect 1-2% ensemble gain')
else:
    print('  ✅ HIGH diversity - expect 2-3% ensemble gain')

# Recommendations
print('\n' + '='*80)
print('💡 RECOMMENDATIONS')
print('='*80)

if best['avg_acc'] >= 0.70:
    print('\n✅ Single model meets target - ready to deploy!')
elif best['avg_acc'] >= 0.68:
    print(f'\n⚠️  Close to target (gap: {(0.70-best["avg_acc"])*100:.2f}%)')
    print('   Options:')
    print('   1. Ensemble top 4-5 models')
    print('   2. Try Test-Time Augmentation')
    print('   3. Tune confidence threshold')
elif best['avg_acc'] >= 0.66:
    print(f'\n❌ Moderate gap to target ({(0.70-best["avg_acc"])*100:.2f}%)')
    print('   Options:')
    print('   1. Ensemble ALL top models')
    print('   2. Accept ~67-68% as realistic for this dataset')
    print('   3. Consider YOLOv11m (if size permits)')
else:
    print(f'\n❌ Significant gap to target ({(0.70-best["avg_acc"])*100:.2f}%)')
    print('   Realistically, 70% may not be achievable with YOLOv11s')
    print('   Consider:')
    print('   1. Accepting current performance')
    print('   2. Improving dataset quality')
    print('   3. Different model architecture')

# Size constraint check
print('\n' + '='*80)
print('📏 SIZE CONSTRAINT CHECK (Target: <70MB)')
print('='*80)

under_limit = [r for r in results if r['size_mb'] < 70]
print(f'\nModels under 70MB: {len(under_limit)}/{len(results)}')

if under_limit:
    best_under_limit = max(under_limit, key=lambda x: x['avg_acc'])
    print(f'\nBest model under 70MB:')
    print(f'  Model: {best_under_limit["name"]}')
    print(f'  Accuracy: {best_under_limit["avg_acc"]*100:.2f}%')
    print(f'  Size: {best_under_limit["size_mb"]:.1f} MB')

print('\n' + '='*80)

# Save results
with open('all_models_ranked.txt', 'w') as f:
    f.write('COMPREHENSIVE MODEL RANKING\n')
    f.write('='*80 + '\n\n')
    for i, r in enumerate(results, 1):
        f.write(f"{i}. {r['name']}\n")
        f.write(f"   Avg Acc: {r['avg_acc']*100:.2f}%  ")
        f.write(f"Precision: {r['precision']*100:.2f}%  ")
        f.write(f"Recall: {r['recall']*100:.2f}%\n")
        f.write(f"   Size: {r['size_mb']:.1f} MB\n\n")

print('\n✓ Full ranking saved to: all_models_ranked.txt')
print('='*80)
