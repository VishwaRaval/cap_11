#!/usr/bin/env python3
"""
Test-Time Augmentation for Ensemble
Expected gain: +1-2% over base ensemble
"""

from ultralytics import YOLO
from pathlib import Path
import numpy as np

print('\n' + '='*80)
print('ENSEMBLE + TEST-TIME AUGMENTATION')
print('='*80)

models_info = [
    {'path': 'runs/detect/extreme_stable_v1/weights/best.pt', 'name': 'extreme_stable_v1'},
    {'path': 'runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt', 'name': 'cosine_finetune'},
    {'path': 'runs/detect/fish_s_s_cosine_ultra_v1/weights/best.pt', 'name': 'cosine_ultra'},
    {'path': 'runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt', 'name': 'recall_optimized'},
]

# Load available models
models = []
for info in models_info:
    if Path(info['path']).exists():
        models.append(YOLO(info['path']))
        print(f"✓ Loaded: {info['name']}")

print(f'\n✓ Loaded {len(models)} models')
print('\nTesting different confidence thresholds with TTA...\n')

best_conf = 0.20
best_avg_acc = 0.0
results_table = []

# Test confidence thresholds with TTA
for conf in [0.15, 0.18, 0.20, 0.22, 0.25]:
    print(f'Testing conf={conf:.2f} with TTA...', end=' ')
    
    # Validate each model with TTA
    precisions = []
    recalls = []
    
    for model in models:
        results = model.val(
            data='dataset_root/data.yaml',
            conf=conf,
            iou=0.5,
            augment=True,  # TTA enabled!
            verbose=False
        )
        
        precisions.append(float(results.box.mp))
        recalls.append(float(results.box.mr))
    
    # Average metrics (ensemble approximation)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_acc = (avg_precision + avg_recall) / 2
    
    results_table.append({
        'conf': conf,
        'precision': avg_precision,
        'recall': avg_recall,
        'avg_acc': avg_acc
    })
    
    status = '✅' if avg_acc >= 0.70 else ''
    print(f'{avg_acc*100:.2f}% {status}')
    
    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        best_conf = conf

# Results
print('\n' + '='*80)
print('ENSEMBLE + TTA RESULTS')
print('='*80)

print(f"\n{'Conf':<8} {'Precision':<12} {'Recall':<12} {'Avg Acc':<12} {'Status':<10}")
print('-'*60)

for r in results_table:
    status = '✅' if r['avg_acc'] >= 0.70 else f"Gap {(0.70-r['avg_acc'])*100:.2f}%"
    print(f"{r['conf']:<8.2f} {r['precision']*100:<11.2f}% {r['recall']*100:<11.2f}% "
          f"{r['avg_acc']*100:<11.2f}% {status}")

print('-'*60)

print(f'\n🎯 BEST RESULT WITH TTA:')
print(f'   Confidence: {best_conf:.2f}')
print(f'   Avg Accuracy: {best_avg_acc*100:.2f}%')
print(f'   Target: 70.00%')

if best_avg_acc >= 0.70:
    print(f'   Status: ✅ TARGET MET WITH TTA!')
    print(f'\n💡 DEPLOYMENT:')
    print(f'   - Use ensemble of {len(models)} models')
    print(f'   - Set conf={best_conf:.2f}')
    print(f'   - Enable TTA: augment=True')
elif best_avg_acc >= 0.69:
    print(f'   Status: ⚠️  VERY CLOSE! Gap: {(0.70-best_avg_acc)*100:.2f}%')
    print(f'\n💡 Almost there! Try:')
    print(f'   - Lower confidence to 0.15-0.18')
    print(f'   - True ensemble + TTA (not just averaging)')
else:
    print(f'   Status: ❌ Gap: {(0.70-best_avg_acc)*100:.2f}%')
    print(f'\n💡 TTA alone not enough. Next:')
    print(f'   - Run true_ensemble_inference.py with augment=True')
    print(f'   - Or train one more model with moderate hyperparameters')

print('='*80)

# Comparison
print('\n📊 IMPROVEMENT BREAKDOWN:')
print('-'*60)
print(f'  Base ensemble (no TTA):  67.19%')
print(f'  Ensemble + TTA:          {best_avg_acc*100:.2f}%')
print(f'  Improvement:             +{(best_avg_acc - 0.6719)*100:.2f}%')
print(f'  Gap to target:           {max(0, (0.70 - best_avg_acc)*100):.2f}%')
print('-'*60)
print('='*80)
