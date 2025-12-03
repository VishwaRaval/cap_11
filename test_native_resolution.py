#!/usr/bin/env python3
"""
Test Existing Models at Native Resolution
Compare performance: 768×768 (square) vs 768×432 (native)
"""

from ultralytics import YOLO
from pathlib import Path

print('\n' + '='*80)
print('NATIVE RESOLUTION TEST - EXISTING MODELS')
print('='*80)
print('\nYour images: 768×432 (landscape)')
print('Current training: 768×768 (square with padding)')
print('\nQuestion: Do models perform better at native resolution?\n')

models_to_test = [
    {
        'path': 'runs/detect/extreme_stable_v1/weights/best.pt',
        'name': 'extreme_stable_v1 (best single)'
    },
    {
        'path': 'runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt', 
        'name': 'cosine_finetune_v1'
    }
]

# Check which models exist
available_models = []
for info in models_to_test:
    if Path(info['path']).exists():
        available_models.append(info)

if not available_models:
    print("❌ No models found!")
    exit(1)

print(f'✓ Found {len(available_models)} models to test\n')

# Test each model at both resolutions
print('='*80)
print('RESOLUTION COMPARISON')
print('='*80)

for info in available_models:
    print(f'\n{"="*80}')
    print(f'Testing: {info["name"]}')
    print(f'{"="*80}')
    
    model = YOLO(info['path'])
    
    # Test 1: Square 768×768 (current)
    print(f'\n1. Square 768×768 (current training size):')
    print('-'*80)
    results_square = model.val(
        data='dataset_root/data.yaml',
        imgsz=768,  # Square
        conf=0.20,
        iou=0.5,
        verbose=False
    )
    
    prec_square = float(results_square.box.mp)
    rec_square = float(results_square.box.mr)
    mAP50_square = float(results_square.box.map50)
    avg_square = (prec_square + rec_square) / 2
    
    print(f'  Precision: {prec_square*100:.2f}%')
    print(f'  Recall:    {rec_square*100:.2f}%')
    print(f'  mAP50:     {mAP50_square*100:.2f}%')
    print(f'  Avg Acc:   {avg_square*100:.2f}%')
    
    # Test 2: Native 768×432
    print(f'\n2. Native 768×432 (original aspect ratio):')
    print('-'*80)
    results_native = model.val(
        data='dataset_root/data.yaml',
        imgsz=[768, 432],  # Native resolution!
        conf=0.20,
        iou=0.5,
        verbose=False
    )
    
    prec_native = float(results_native.box.mp)
    rec_native = float(results_native.box.mr)
    mAP50_native = float(results_native.box.map50)
    avg_native = (prec_native + rec_native) / 2
    
    print(f'  Precision: {prec_native*100:.2f}%')
    print(f'  Recall:    {rec_native*100:.2f}%')
    print(f'  mAP50:     {mAP50_native*100:.2f}%')
    print(f'  Avg Acc:   {avg_native*100:.2f}%')
    
    # Comparison
    print(f'\n📊 Difference (Native - Square):')
    print('-'*80)
    print(f'  Precision: {(prec_native - prec_square)*100:+.2f}%')
    print(f'  Recall:    {(rec_native - rec_square)*100:+.2f}%')
    print(f'  mAP50:     {(mAP50_native - mAP50_square)*100:+.2f}%')
    print(f'  Avg Acc:   {(avg_native - avg_square)*100:+.2f}%')
    
    if avg_native > avg_square:
        print(f'\n  ✅ Native resolution is BETTER by {(avg_native - avg_square)*100:.2f}%!')
    elif avg_native < avg_square:
        print(f'\n  ⚠️  Square resolution is better by {(avg_square - avg_native)*100:.2f}%')
    else:
        print(f'\n  ➡️  No significant difference')

print('\n' + '='*80)
print('SUMMARY & RECOMMENDATION')
print('='*80)
print('\nIf native resolution shows improvement:')
print('  → Cancel current training')
print('  → Retrain at 768×432 using train_native_resolution.py')
print('\nIf square resolution is better or same:')
print('  → Keep current training')
print('  → Models already optimal at 768×768')
print('='*80)
