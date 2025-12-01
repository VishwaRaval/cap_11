#!/usr/bin/env python3
"""
Check class distribution in YOLO format dataset.
Identifies class imbalance issues.
"""

import sys
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def count_classes_in_labels(labels_dir, class_names):
    """Count instances of each class in label files."""
    labels_dir = Path(labels_dir)
    
    class_counts = Counter()
    total_images = 0
    images_per_class = {i: set() for i in range(len(class_names))}
    
    for label_file in labels_dir.glob('*.txt'):
        total_images += 1
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    images_per_class[class_id].add(label_file.stem)
    
    return class_counts, total_images, images_per_class


def analyze_dataset(dataset_root):
    """Analyze class distribution across train/val/test splits."""
    dataset_root = Path(dataset_root)
    
    class_names = ['Grunt Fish', 'Parrot Fish', 'Surgeon Fish']
    
    print("=" * 70)
    print("CLASS DISTRIBUTION ANALYSIS - 3-Class Dataset")
    print("=" * 70)
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        labels_dir = dataset_root / split / 'labels'
        
        if not labels_dir.exists():
            print(f"\n⚠️  {split.upper()} labels not found at: {labels_dir}")
            continue
        
        counts, total_images, images_per_class = count_classes_in_labels(labels_dir, class_names)
        
        print(f"\n📊 {split.upper()} SET:")
        print(f"  Total images: {total_images}")
        print(f"\n  Class distribution:")
        
        total_instances = sum(counts.values())
        
        for class_id in range(len(class_names)):
            count = counts.get(class_id, 0)
            percentage = (count / total_instances * 100) if total_instances > 0 else 0
            img_count = len(images_per_class[class_id])
            img_percentage = (img_count / total_images * 100) if total_images > 0 else 0
            
            bar = '█' * int(percentage / 2)
            print(f"  {class_names[class_id]:15s}: {count:5d} instances ({percentage:5.1f}%) "
                  f"in {img_count:4d} images ({img_percentage:5.1f}%) {bar}")
        
        # Check for imbalance
        if total_instances > 0:
            max_count = max(counts.values())
            min_count = min(counts.values()) if counts else 0
            
            if min_count > 0:
                imbalance_ratio = max_count / min_count
                print(f"\n  Imbalance ratio: {imbalance_ratio:.2f}x")
                
                if imbalance_ratio > 3:
                    print(f"  ❌ SEVERE CLASS IMBALANCE (ratio > 3)")
                    print(f"     Consider class weighting or oversampling minority classes")
                elif imbalance_ratio > 2:
                    print(f"  ⚠️  MODERATE CLASS IMBALANCE (ratio > 2)")
                else:
                    print(f"  ✓ Balanced classes")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    
    print("\n1. If class imbalance exists:")
    print("   - Use class weights in loss function")
    print("   - Oversample minority classes during training")
    print("   - Use focal loss to focus on hard examples")
    
    print("\n2. For multi-class detection:")
    print("   - Increase cls_loss weight (currently 0.5, try 0.7-1.0)")
    print("   - Train longer (150-200 epochs for 3 classes)")
    print("   - Monitor per-class metrics in W&B")
    
    print("\n3. Expected metrics for 3-class:")
    print("   - mAP@50: 55-70% (lower than single-class)")
    print("   - Recall per class: Monitor separately")
    print("   - Some classes may perform better than others")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
    else:
        dataset_root = 'dataset_root'
    
    analyze_dataset(dataset_root)
