#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Validation
Applies augmentations during inference for +1-2% boost
"""

from ultralytics import YOLO

# Load your best model
model = YOLO('runs/detect/extreme_stable_v1/weights/best.pt')

print('\n' + '='*80)
print('VALIDATING WITH TEST-TIME AUGMENTATION (TTA)')
print('='*80)
print('\nThis applies augmentations during inference for +1-2% boost')
print('Augmentations: horizontal flip, slight scale variations\n')

# Validate with TTA
results = model.val(
    data='dataset_root/data.yaml',
    augment=True,      # Enable TTA
    conf=0.20,
    iou=0.5,
    verbose=True
)

# Extract metrics
precision = float(results.box.mp)
recall = float(results.box.mr)
mAP50 = float(results.box.map50)
mAP50_95 = float(results.box.map)
avg_acc = (precision + recall) / 2
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print('\n' + '='*80)
print('RESULTS WITH TTA')
print('='*80)
print(f'\nMetrics:')
print(f'  Precision:   {precision:.4f} ({precision*100:.2f}%)')
print(f'  Recall:      {recall:.4f} ({recall*100:.2f}%)')
print(f'  F1 Score:    {f1:.4f} ({f1*100:.2f}%)')
print(f'  mAP50:       {mAP50:.4f} ({mAP50*100:.2f}%)')
print(f'  mAP50-95:    {mAP50_95:.4f} ({mAP50_95*100:.2f}%)')
print(f'  Avg Acc:     {avg_acc:.4f} ({avg_acc*100:.2f}%)')

print(f'\nTarget: 70.00%')
print(f'Gap: {max(0, 0.70-avg_acc):.4f} ({max(0, (0.70-avg_acc)*100):.2f}%)')

if avg_acc >= 0.70:
    print('\n✅ TARGET MET WITH TTA!')
    print('   Deploy this model with TTA enabled')
else:
    print(f'\n⚠️  Still {(0.70-avg_acc)*100:.2f}% below target')
    print('   Try confidence threshold tuning next')

print('='*80)
print(f'\nModel: extreme_stable_v1')
print(f'Size: ~47MB (under 75MB limit)')
print(f'Deployment: Single model with augment=True')
print('='*80)
