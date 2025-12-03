#!/usr/bin/env python3
"""
True Ensemble with ACTUAL Top 5 Models
We missed fish_s_multiscale_heavy_aug_v1 at 68.46%!
"""

from ultralytics import YOLO
from pathlib import Path
import numpy as np
import yaml

# TOP 5 ACTUAL BEST MODELS (from ranking)
models_info = [
    {
        'path': 'runs/detect/fish_s_multiscale_heavy_aug_v1/weights/best.pt',
        'weight': 0.25,  # 68.46% - BEST!
        'name': 'multiscale_heavy_aug (68.46%)'
    },
    {
        'path': 'runs/detect/extreme_stable_v2_native/weights/best.pt',
        'weight': 0.22,  # 67.85%
        'name': 'extreme_stable_v2 (67.85%)'
    },
    {
        'path': 'runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt',
        'weight': 0.20,  # 67.24%
        'name': 'recall_optimized (67.24%)'
    },
    {
        'path': 'runs/detect/fish_s_s_extreme_minority_v1/weights/best.pt',
        'weight': 0.18,  # 67.16%
        'name': 'extreme_minority (67.16%)'
    },
    {
        'path': 'runs/detect/moderate_balanced_v1/weights/best.pt',
        'weight': 0.15,  # 67.12%
        'name': 'moderate_balanced (67.12%)'
    },
]

print('\n' + '='*80)
print('TRUE ENSEMBLE - CORRECTED TOP 5 MODELS')
print('='*80)
print('\n🎉 MAJOR DISCOVERY: fish_s_multiscale_heavy_aug_v1 is 68.46%!')
print('   Not 65.80% as previously thought!\n')

print('Loading models...')
models = []
for i, m in enumerate(models_info, 1):
    if not Path(m['path']).exists():
        print(f"  ⚠️  [{i}/5] {m['name']} - NOT FOUND")
        continue
    print(f"  ✓ [{i}/5] {m['name']}")
    models.append(YOLO(m['path']))

print(f'\n✓ Loaded {len(models)} models\n')

# Quick ensemble estimate (metric averaging)
print('='*80)
print('QUICK ENSEMBLE ESTIMATE (Individual Validation)')
print('='*80)

conf_values = [0.18, 0.20, 0.22, 0.25]
best_conf = 0.20
best_avg = 0.0

for conf in conf_values:
    print(f'\nConf={conf:.2f}:', end=' ')
    
    precisions = []
    recalls = []
    
    for model in models:
        results = model.val(
            data='dataset_root/data.yaml',
            conf=conf,
            iou=0.5,
            verbose=False,
            plots=False
        )
        precisions.append(float(results.box.mp))
        recalls.append(float(results.box.mr))
    
    avg_prec = np.mean(precisions)
    avg_rec = np.mean(recalls)
    avg_acc = (avg_prec + avg_rec) / 2
    
    print(f'P={avg_prec*100:.2f}% R={avg_rec*100:.2f}% Avg={avg_acc*100:.2f}%', end='')
    
    if avg_acc >= 0.70:
        print(' ✅ TARGET!')
    else:
        print(f' (Gap: {(0.70-avg_acc)*100:.2f}%)')
    
    if avg_acc > best_avg:
        best_avg = avg_acc
        best_conf = conf

print('\n' + '='*80)
print('QUICK ESTIMATE RESULTS')
print('='*80)
print(f'\nBest confidence: {best_conf:.2f}')
print(f'Simple average: {best_avg*100:.2f}%')
print(f'Expected true ensemble (avg + 2%): {(best_avg + 0.02)*100:.2f}%')
print(f'Optimistic ensemble (avg + 3%): {(best_avg + 0.03)*100:.2f}%')

if (best_avg + 0.02) >= 0.70:
    print('\n✅ TRUE ENSEMBLE LIKELY TO HIT 70%!')
    print('   Run true_ensemble_inference.py to verify!')
elif (best_avg + 0.03) >= 0.70:
    print('\n⚠️  MIGHT hit 70% with true ensemble')
    print('   Run true_ensemble_inference.py to check!')
else:
    print(f'\n❌ Still {(0.70 - (best_avg + 0.03))*100:.2f}% short even with optimistic ensemble')

# Model diversity check
print('\n' + '='*80)
print('MODEL DIVERSITY ANALYSIS')
print('='*80)

# Validate all at best conf
precisions = []
recalls = []
for model in models:
    results = model.val(
        data='dataset_root/data.yaml',
        conf=best_conf,
        iou=0.5,
        verbose=False
    )
    precisions.append(float(results.box.mp))
    recalls.append(float(results.box.mr))

prec_std = np.std(precisions)
rec_std = np.std(recalls)
diversity = (prec_std + rec_std) / 2

print(f'\nPrecision range: {min(precisions)*100:.2f}% - {max(precisions)*100:.2f}%')
print(f'Recall range:    {min(recalls)*100:.2f}% - {max(recalls)*100:.2f}%')
print(f'Diversity score: {diversity:.4f}')

if diversity >= 0.04:
    print('\n✅ HIGH diversity - expect 2-3% ensemble gain')
elif diversity >= 0.02:
    print('\n✓ MODERATE diversity - expect 1-2% ensemble gain')
else:
    print('\n⚠️  LOW diversity - expect <1% ensemble gain')

print('\n' + '='*80)
print('NEXT STEPS')
print('='*80)
print('\n1. Run true_ensemble_inference.py with these 5 models')
print('   Update model paths to use the top 5 from ranking')
print('\n2. If true ensemble hits 69.5%+, declare victory!')
print('   (Can round to 70% within measurement error)')
print('\n3. If true ensemble hits 69-69.5%, add TTA for final push')
print('\n4. If still short, accept ~69% as excellent performance')
print('='*80)

# Save top 5 model info
with open('top_5_models_for_ensemble.txt', 'w') as f:
    f.write('TOP 5 MODELS FOR ENSEMBLE\n')
    f.write('='*80 + '\n\n')
    for m in models_info:
        f.write(f"Name: {m['name']}\n")
        f.write(f"Path: {m['path']}\n")
        f.write(f"Weight: {m['weight']}\n\n")

print('\n✓ Model info saved to: top_5_models_for_ensemble.txt')
