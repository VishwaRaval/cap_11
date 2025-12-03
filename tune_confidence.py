#!/usr/bin/env python3
"""
Confidence Threshold Tuning
Tests different confidence thresholds to find optimal precision/recall balance
"""

from ultralytics import YOLO

# Load your best model
model = YOLO('runs/detect/extreme_stable_v1/weights/best.pt')

print('\n' + '='*80)
print('TESTING DIFFERENT CONFIDENCE THRESHOLDS')
print('='*80)
print('\nFinding optimal threshold for 70% target\n')

best_conf = 0.20
best_avg_acc = 0.0
results_table = []

# Test different confidence thresholds
conf_values = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]

for conf in conf_values:
    print(f'Testing confidence: {conf:.2f}...', end=' ')
    
    results = model.val(
        data='dataset_root/data.yaml',
        conf=conf,
        iou=0.5,
        verbose=False,
        plots=False
    )
    
    precision = float(results.box.mp)
    recall = float(results.box.mr)
    mAP50 = float(results.box.map50)
    avg_acc = (precision + recall) / 2
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    results_table.append({
        'conf': conf,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP50': mAP50,
        'avg_acc': avg_acc
    })
    
    status = '✅' if avg_acc >= 0.70 else ''
    print(f'Avg Acc: {avg_acc*100:.2f}% {status}')
    
    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        best_conf = conf

# Print results table
print('\n' + '='*80)
print('CONFIDENCE THRESHOLD COMPARISON')
print('='*80)
print(f"\n{'Conf':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'mAP50':<12} {'Avg Acc':<12} {'Target':<8}")
print('-'*80)

for r in results_table:
    status = '✅' if r['avg_acc'] >= 0.70 else ''
    print(f"{r['conf']:<8.2f} {r['precision']*100:<11.2f}% {r['recall']*100:<11.2f}% "
          f"{r['f1']*100:<11.2f}% {r['mAP50']*100:<11.2f}% {r['avg_acc']*100:<11.2f}% {status}")

print('-'*80)

# Find best threshold
print(f'\n🎯 OPTIMAL THRESHOLD: {best_conf:.2f}')
print(f'   Best Avg Accuracy: {best_avg_acc*100:.2f}%')
print(f'   Target: 70.00%')

if best_avg_acc >= 0.70:
    print(f'   Status: ✅ TARGET MET!')
    print(f'\n   Deploy with conf={best_conf:.2f}')
else:
    print(f'   Status: ❌ Gap: {(0.70-best_avg_acc)*100:.2f}%')
    print(f'\n   Best achievable with threshold tuning: {best_avg_acc*100:.2f}%')
    print(f'   Try TTA with optimal threshold next')

print('='*80)

# Show precision-recall tradeoff
print('\n📊 PRECISION-RECALL TRADEOFF:')
print('  Lower conf → Higher recall, Lower precision')
print('  Higher conf → Lower recall, Higher precision')
print(f'\n  Your best balance: conf={best_conf:.2f}')
print('='*80)
