#!/usr/bin/env python3
"""
Quick Ensemble Reality Check
Compares individual model metrics vs simple ensemble averaging
"""

from ultralytics import YOLO
from pathlib import Path

print('\n' + '='*80)
print('ENSEMBLE REALITY CHECK')
print('='*80)

models_info = [
    {'path': 'runs/detect/extreme_stable_v1/weights/best.pt', 'name': 'extreme_stable_v1'},
    {'path': 'runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt', 'name': 'cosine_finetune'},
    {'path': 'runs/detect/fish_s_s_cosine_ultra_v1/weights/best.pt', 'name': 'cosine_ultra'},
    {'path': 'runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt', 'name': 'recall_optimized'},
]

# Check which models exist
available_models = []
for info in models_info:
    if Path(info['path']).exists():
        available_models.append(info)
    else:
        print(f"⚠️  Model not found: {info['name']}")

if not available_models:
    print("❌ No models found!")
    exit(1)

print(f'\n✓ Found {len(available_models)} models\n')

# Validate each individually
print('='*80)
print('INDIVIDUAL MODEL VALIDATION')
print('='*80)

individual_results = []
for info in available_models:
    print(f'\nValidating: {info["name"]}')
    print('-'*80)
    
    model = YOLO(info['path'])
    results = model.val(
        data='dataset_root/data.yaml',
        conf=0.25,
        iou=0.5,
        verbose=False
    )
    
    precision = float(results.box.mp)
    recall = float(results.box.mr)
    mAP50 = float(results.box.map50)
    avg_acc = (precision + recall) / 2
    
    individual_results.append({
        'name': info['name'],
        'precision': precision,
        'recall': recall,
        'mAP50': mAP50,
        'avg_acc': avg_acc
    })
    
    print(f'  Precision: {precision*100:.2f}%')
    print(f'  Recall:    {recall*100:.2f}%')
    print(f'  mAP50:     {mAP50*100:.2f}%')
    print(f'  Avg Acc:   {avg_acc*100:.2f}%')

# Calculate simple average (upper bound estimate)
print('\n' + '='*80)
print('ENSEMBLE ESTIMATES')
print('='*80)

avg_precision = sum(r['precision'] for r in individual_results) / len(individual_results)
avg_recall = sum(r['recall'] for r in individual_results) / len(individual_results)
avg_mAP50 = sum(r['mAP50'] for r in individual_results) / len(individual_results)
simple_avg = (avg_precision + avg_recall) / 2

print(f'\n1. Simple Average (what your script did):')
print(f'   Precision: {avg_precision*100:.2f}%')
print(f'   Recall:    {avg_recall*100:.2f}%')
print(f'   Avg Acc:   {simple_avg*100:.2f}%')

print(f'\n2. With 3.5% ensemble boost (your final_optimization.py):')
boosted_precision = min(avg_precision + 0.035, 1.0)
boosted_recall = min(avg_recall + 0.035, 1.0)
boosted_avg = (boosted_precision + boosted_recall) / 2
print(f'   Precision: {boosted_precision*100:.2f}%')
print(f'   Recall:    {boosted_recall*100:.2f}%')
print(f'   Avg Acc:   {boosted_avg*100:.2f}% ← Your reported 70.37%')

print(f'\n3. Realistic ensemble estimate (+2-3% typical gain):')
realistic_precision = min(avg_precision + 0.025, 1.0)
realistic_recall = min(avg_recall + 0.025, 1.0)
realistic_avg = (realistic_precision + realistic_recall) / 2
print(f'   Precision: {realistic_precision*100:.2f}%')
print(f'   Recall:    {realistic_recall*100:.2f}%')
print(f'   Avg Acc:   {realistic_avg*100:.2f}%')

print('\n' + '='*80)
print('REALITY CHECK SUMMARY')
print('='*80)

best_single = max(r['avg_acc'] for r in individual_results)
print(f'\nBest single model:     {best_single*100:.2f}%')
print(f'Simple average:        {simple_avg*100:.2f}%')
print(f'Your estimate (3.5%):  {boosted_avg*100:.2f}%')
print(f'Realistic (+2.5%):     {realistic_avg*100:.2f}%')
print(f'Target:                70.00%')

if realistic_avg >= 0.70:
    print('\n✅ Realistic ensemble likely meets target!')
elif realistic_avg >= 0.68:
    gap = (0.70 - realistic_avg) * 100
    print(f'\n⚠️  Close! Gap: {gap:.2f}%')
    print('   Add TTA for +1-2% to reach target')
else:
    gap = (0.70 - realistic_avg) * 100
    print(f'\n❌ Significant gap: {gap:.2f}%')
    print('   Need better base models or different approach')

print('\n' + '='*80)
print('NEXT STEPS')
print('='*80)
print('\n1. Run true_ensemble_inference.py to get ACTUAL accuracy')
print('   This does proper box-level fusion, not just metric averaging')
print('\n2. If true ensemble < 70%, try:')
print('   - Add Test-Time Augmentation (TTA)')
print('   - Tune confidence threshold')
print('   - Train one more model with moderate hyperparameters')
print('\n3. Only add to results table AFTER running true_ensemble_inference.py')
print('='*80)
