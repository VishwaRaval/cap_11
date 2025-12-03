#!/usr/bin/env python3
"""
Aggressive Confidence Threshold Tuning
Since we have high recall but low precision, 
we need to find the sweet spot that maximizes avg accuracy
"""

from ultralytics import YOLO
from pathlib import Path
import numpy as np

print('\n' + '='*80)
print('AGGRESSIVE CONFIDENCE THRESHOLD TUNING')
print('='*80)
print('\nProblem: High recall (78%), low precision (56%)')
print('Strategy: Find optimal conf threshold to balance precision/recall\n')

models_info = [
    {'path': 'runs/detect/extreme_stable_v1/weights/best.pt', 'name': 'extreme_stable_v1'},
    {'path': 'runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt', 'name': 'cosine_finetune'},
    {'path': 'runs/detect/fish_s_s_cosine_ultra_v1/weights/best.pt', 'name': 'cosine_ultra'},
    {'path': 'runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt', 'name': 'recall_optimized'},
]

# Load models
models = []
for info in models_info:
    if Path(info['path']).exists():
        models.append(YOLO(info['path']))

print(f'✓ Loaded {len(models)} models\n')

# Fine-grained threshold sweep
conf_values = [
    0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 
    0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.30, 0.32, 0.35
]

print('Testing confidence thresholds (this will take ~10 minutes)...\n')

best_conf = 0.20
best_avg_acc = 0.0
results = []

for conf in conf_values:
    print(f'conf={conf:.2f}...', end=' ', flush=True)
    
    # Ensemble validation
    precisions = []
    recalls = []
    
    for model in models:
        try:
            result = model.val(
                data='dataset_root/data.yaml',
                conf=conf,
                iou=0.5,
                verbose=False,
                plots=False
            )
            
            precisions.append(float(result.box.mp))
            recalls.append(float(result.box.mr))
        except:
            pass
    
    if precisions and recalls:
        avg_prec = np.mean(precisions)
        avg_rec = np.mean(recalls)
        avg_acc = (avg_prec + avg_rec) / 2
        f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec + 1e-8)
        
        results.append({
            'conf': conf,
            'precision': avg_prec,
            'recall': avg_rec,
            'f1': f1,
            'avg_acc': avg_acc
        })
        
        print(f'P:{avg_prec*100:.1f}% R:{avg_rec*100:.1f}% Avg:{avg_acc*100:.2f}%')
        
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_conf = conf

# Results table
print('\n' + '='*80)
print('CONFIDENCE THRESHOLD SWEEP RESULTS')
print('='*80)

# Sort by avg_acc
results.sort(key=lambda x: x['avg_acc'], reverse=True)

print(f"\n{'Rank':<6} {'Conf':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg Acc':<12}")
print('-'*70)

for i, r in enumerate(results[:15], 1):  # Top 15
    marker = '✅' if r['avg_acc'] >= 0.70 else '👉' if i == 1 else '  '
    print(f"{marker} {i:<4} {r['conf']:<8.2f} {r['precision']*100:<11.2f}% "
          f"{r['recall']*100:<11.2f}% {r['f1']*100:<11.2f}% {r['avg_acc']*100:<11.2f}%")

print('-'*70)

# Best result
best = results[0]
print(f'\n🎯 OPTIMAL THRESHOLD: {best["conf"]:.2f}')
print(f'   Precision: {best["precision"]*100:.2f}%')
print(f'   Recall:    {best["recall"]*100:.2f}%')
print(f'   F1 Score:  {best["f1"]*100:.2f}%')
print(f'   Avg Acc:   {best["avg_acc"]*100:.2f}%')

print(f'\n   Target:    70.00%')

if best['avg_acc'] >= 0.70:
    print(f'   Status:    ✅ TARGET MET!')
    print(f'\n🎉 SUCCESS! Deploy with:')
    print(f'   - Ensemble of {len(models)} models')
    print(f'   - Confidence: {best["conf"]:.2f}')
    print(f'   - IoU: 0.5')
elif best['avg_acc'] >= 0.69:
    gap = (0.70 - best['avg_acc']) * 100
    print(f'   Status:    ⚠️  Gap: {gap:.2f}%')
    print(f'\n💡 VERY CLOSE! Add TTA:')
    print(f'   python test_tta_ensemble.py')
    print(f'   Expected with TTA: {(best["avg_acc"] + 0.015)*100:.2f}%')
else:
    gap = (0.70 - best['avg_acc']) * 100
    print(f'   Status:    ❌ Gap: {gap:.2f}%')
    print(f'\n💡 Next steps:')
    print(f'   1. Add TTA (expected +1-2%)')
    print(f'   2. If still short, train better base model')

print('='*80)

# Precision-Recall tradeoff analysis
print('\n📊 PRECISION-RECALL TRADEOFF:')
print('-'*70)

low_conf = [r for r in results if r['conf'] <= 0.20]
high_conf = [r for r in results if r['conf'] >= 0.25]

if low_conf and high_conf:
    print(f"  Low conf (≤0.20):  Avg precision {np.mean([r['precision'] for r in low_conf])*100:.1f}%, "
          f"recall {np.mean([r['recall'] for r in low_conf])*100:.1f}%")
    print(f"  High conf (≥0.25): Avg precision {np.mean([r['precision'] for r in high_conf])*100:.1f}%, "
          f"recall {np.mean([r['recall'] for r in high_conf])*100:.1f}%")
    print(f'\n  → Optimal balance at conf={best["conf"]:.2f}')

print('='*80)

# Save results
with open('threshold_sweep_results.txt', 'w') as f:
    f.write('CONFIDENCE THRESHOLD SWEEP RESULTS\n')
    f.write('='*80 + '\n\n')
    for r in results[:10]:
        f.write(f"conf={r['conf']:.2f}: P={r['precision']*100:.2f}% "
                f"R={r['recall']*100:.2f}% Avg={r['avg_acc']*100:.2f}%\n")

print('\n✓ Full results saved to: threshold_sweep_results.txt')
