#!/usr/bin/env python3
"""
Generate Predictions on Entire Dataset
Uses top 3 single models to create bounding box predictions
Saves predictions in YOLO format for all images in train/valid/test
"""

from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import shutil

print('\n' + '='*80)
print('GENERATE PREDICTIONS - ENTIRE DATASET')
print('='*80)
print('\nGenerating bounding boxes for all images (train + valid + test)\n')

# Top 3 models
models_to_test = [
    {
        'path': 'runs/detect/fish_s_multiscale_heavy_aug_v1/weights/best.pt',
        'name': 'multiscale_heavy_aug',
        'acc': 68.46
    },
    {
        'path': 'runs/detect/extreme_stable_v2_native/weights/best.pt',
        'name': 'extreme_stable_v2',
        'acc': 67.85
    },
    {
        'path': 'runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt',
        'name': 'recall_optimized',
        'acc': 67.24
    },
]

# Dataset paths
dataset_root = Path('dataset_root')
splits = ['train', 'valid', 'test']

# Output directory
output_root = Path('predictions_all_dataset')
output_root.mkdir(exist_ok=True)

print('Dataset splits to process:')
for split in splits:
    images_path = dataset_root / split / 'images'
    n_images = len(list(images_path.glob('*.jpg'))) + len(list(images_path.glob('*.png')))
    print(f'  {split:8s}: {n_images:4d} images')

print(f'\nOutput directory: {output_root}')
print()

# Process each model
for model_info in models_to_test:
    print('='*80)
    print(f'MODEL: {model_info["name"]} ({model_info["acc"]:.2f}%)')
    print('='*80)
    
    # Check if model exists
    if not Path(model_info['path']).exists():
        print(f'❌ Model not found: {model_info["path"]}')
        continue
    
    # Load model
    print(f'\nLoading model...', end=' ')
    model = YOLO(model_info['path'])
    print('✓')
    
    # Create output directory for this model
    model_output = output_root / model_info['name']
    model_output.mkdir(exist_ok=True)
    
    # Process each split
    for split in splits:
        print(f'\n{split.upper()} SET:')
        print('-'*80)
        
        # Get images
        images_path = dataset_root / split / 'images'
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        if not image_files:
            print(f'  No images found in {images_path}')
            continue
        
        # Create output directories
        split_output = model_output / split
        images_output = split_output / 'images'
        labels_output = split_output / 'labels'
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)
        
        # Run predictions
        print(f'  Processing {len(image_files)} images...')
        
        for img_path in tqdm(image_files, desc=f'  Predicting'):
            # Run inference
            results = model.predict(
                source=str(img_path),
                conf=0.20,  # Standard confidence
                iou=0.5,
                verbose=False,
                save=False
            )
            
            # Copy image
            shutil.copy(img_path, images_output / img_path.name)
            
            # Save predictions in YOLO format
            label_file = labels_output / (img_path.stem + '.txt')
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                result = results[0]
                boxes = result.boxes.xywhn.cpu().numpy()  # normalized xywh
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                with open(label_file, 'w') as f:
                    for box, cls, conf in zip(boxes, classes, confidences):
                        # YOLO format: class x_center y_center width height
                        f.write(f'{int(cls)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n')
            else:
                # No detections - create empty file
                label_file.touch()
        
        print(f'  ✓ Saved to: {split_output}')
    
    print(f'\n✓ Completed predictions for {model_info["name"]}')

# Summary
print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f'\nPredictions saved to: {output_root}')
print('\nDirectory structure:')
print(f'{output_root}/')
for model_info in models_to_test:
    if Path(model_info['path']).exists():
        print(f'  ├── {model_info["name"]}/')
        for split in splits:
            print(f'  │   ├── {split}/')
            print(f'  │   │   ├── images/  (copied original images)')
            print(f'  │   │   └── labels/  (predicted bounding boxes)')

print('\n' + '='*80)
print('PREDICTION FORMAT')
print('='*80)
print('\nEach label file contains predictions in YOLO format:')
print('  class_id x_center y_center width height')
print('  (all values normalized to 0-1)')
print('\nClass IDs:')
print('  0 = Grunt Fish')
print('  1 = Parrot Fish')
print('  2 = Surgeon Fish')

# Statistics
print('\n' + '='*80)
print('STATISTICS')
print('='*80)

for model_info in models_to_test:
    model_output = output_root / model_info['name']
    if not model_output.exists():
        continue
    
    print(f'\n{model_info["name"]} ({model_info["acc"]:.2f}%):')
    
    total_detections = 0
    for split in splits:
        labels_path = model_output / split / 'labels'
        if not labels_path.exists():
            continue
        
        n_files = 0
        n_detections = 0
        
        for label_file in labels_path.glob('*.txt'):
            n_files += 1
            with open(label_file, 'r') as f:
                lines = f.readlines()
                n_detections += len(lines)
        
        print(f'  {split:8s}: {n_files:4d} images, {n_detections:5d} detections')
        total_detections += n_detections
    
    print(f'  {"TOTAL":8s}: {total_detections:5d} detections')

print('\n' + '='*80)
print('✓ COMPLETE')
print('='*80)
print(f'\nAll predictions saved to: {output_root}/')
print('\nYou can now:')
print('  1. Visualize predictions using visualization scripts')
print('  2. Compare predictions between models')
print('  3. Use predictions for further analysis')
print('='*80)
