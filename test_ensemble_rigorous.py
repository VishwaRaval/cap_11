#!/usr/bin/env python3
"""
Test ensemble with rigorous IoU-based evaluation
This is the REAL test of your ensemble results
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ensemble_predictor import FishEnsemble
import numpy as np
from tqdm import tqdm
import json
import cv2


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def yolo_to_xyxy(yolo_box, img_width, img_height):
    """Convert YOLO format [x_center, y_center, width, height] to [x1, y1, x2, y2]"""
    x_center, y_center, width, height = yolo_box
    
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    
    return [x1, y1, x2, y2]


def test_ensemble_rigorous(model_paths, image_dir, labels_dir, 
                          weights=None, method='wbf',
                          conf_threshold=0.25, iou_threshold=0.5):
    """
    Test ensemble with proper IoU-based matching
    """
    print(f"\n{'='*80}")
    print("RIGOROUS ENSEMBLE EVALUATION")
    print(f"{'='*80}")
    print(f"Method: {method}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Models: {len(model_paths)}")
    
    # Create ensemble
    ensemble = FishEnsemble(
        model_paths=model_paths,
        weights=weights,
        conf_threshold=conf_threshold,
        iou_threshold=0.45  # NMS threshold
    )
    
    image_dir = Path(image_dir)
    labels_dir = Path(labels_dir)
    
    # Get all images
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    if len(image_files) == 0:
        print(f"\n❌ ERROR: No images found in {image_dir}")
        print(f"   Please check the path!")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    # Track metrics
    class_tp = {0: 0, 1: 0, 2: 0}
    class_fp = {0: 0, 1: 0, 2: 0}
    class_fn = {0: 0, 1: 0, 2: 0}
    class_gt_count = {0: 0, 1: 0, 2: 0}
    
    processed_images = 0
    
    print(f"\nProcessing images...")
    
    for img_path in tqdm(image_files):
        # Get ground truth
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        
        # Read image for dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        
        # Parse ground truth
        gt_boxes = []
        gt_classes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    yolo_box = [float(x) for x in parts[1:5]]
                    xyxy = yolo_to_xyxy(yolo_box, img_width, img_height)
                    
                    gt_boxes.append(xyxy)
                    gt_classes.append(cls)
                    class_gt_count[cls] += 1
        
        if len(gt_boxes) == 0:
            continue
        
        # Get ensemble predictions
        pred = ensemble.predict(str(img_path), method=method)
        pred_boxes = pred['boxes']
        pred_classes = pred['classes'].astype(int)
        
        # Match predictions to ground truth using IoU
        gt_matched = set()
        
        for pred_idx in range(len(pred_boxes)):
            pred_box = pred_boxes[pred_idx]
            pred_cls = pred_classes[pred_idx]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in gt_matched:
                    continue
                
                gt_box = gt_boxes[gt_idx]
                gt_cls = gt_classes[gt_idx]
                
                if pred_cls != gt_cls:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                class_tp[pred_cls] += 1
                gt_matched.add(best_gt_idx)
            else:
                class_fp[pred_cls] += 1
        
        # False negatives
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in gt_matched:
                class_fn[gt_classes[gt_idx]] += 1
        
        processed_images += 1
    
    if processed_images == 0:
        print(f"\n❌ ERROR: No images were processed!")
        print(f"   Check that labels exist in {labels_dir}")
        return
    
    # Calculate metrics
    class_names = ['Grunt Fish', 'Parrot Fish', 'Surgeon Fish']
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Processed: {processed_images} images")
    
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    total_gt = sum(class_gt_count.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"\nOverall Metrics (IoU >= {iou_threshold}):")
    print(f"  Precision: {overall_precision*100:.2f}%")
    print(f"  Recall:    {overall_recall*100:.2f}%")
    print(f"  F1 Score:  {overall_f1*100:.2f}%")
    print(f"\n  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Ground Truth:    {total_gt}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("  " + "-" * 76)
    
    for cls, class_name in enumerate(class_names):
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        print(f"  {class_name:<15} {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}% {tp:>6d} {fp:>6d} {fn:>6d}")
    
    # Check if 70% target achieved
    print(f"\n{'='*80}")
    if overall_recall >= 0.70:
        print(f"🎉 SUCCESS! Achieved 70% recall target with rigorous IoU matching!")
        print(f"   Recall: {overall_recall*100:.2f}%")
    else:
        shortfall = (0.70 - overall_recall) * 100
        print(f"📊 Current recall: {overall_recall*100:.2f}%")
        print(f"   Need {shortfall:.2f}% more for 70% target")
    print(f"{'='*80}")
    
    # Save results
    results = {
        'method': method,
        'iou_threshold': iou_threshold,
        'conf_threshold': conf_threshold,
        'processed_images': processed_images,
        'overall': {
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1': float(overall_f1),
            'tp': int(total_tp),
            'fp': int(total_fp),
            'fn': int(total_fn)
        },
        'per_class': {}
    }
    
    for cls, class_name in enumerate(class_names):
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        results['per_class'][class_name] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    with open('ensemble_rigorous_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: ensemble_rigorous_results.json")
    
    return results


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Rigorous ensemble evaluation with IoU matching')
    parser.add_argument('--models', nargs='+', required=True, help='Model .pt files')
    
    # Support both --data (YAML) and --images/--labels (directories)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data', help='Path to data.yaml file')
    data_group.add_argument('--images', help='Path to test images directory')
    
    parser.add_argument('--labels', help='Path to test labels directory (required if using --images)')
    parser.add_argument('--weights', nargs='+', type=float, default=None, help='Model weights')
    parser.add_argument('--method', choices=['wbf', 'voting'], default='wbf', help='Ensemble method')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for matching')
    
    args = parser.parse_args()
    
    # Parse data paths
    if args.data:
        print(f"Loading dataset info from: {args.data}")
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        
        yaml_dir = Path(args.data).parent
        
        # Try to find test/val split
        if 'test' in data_config:
            dataset_path = data_config['test']
        elif 'val' in data_config:
            dataset_path = data_config['val']
        else:
            raise ValueError("data.yaml must contain 'test' or 'val' key")
        
        # Handle relative paths
        if not Path(dataset_path).is_absolute():
            dataset_path = yaml_dir / dataset_path
        
        # Assume standard YOLO structure
        if Path(dataset_path).is_dir():
            if (Path(dataset_path) / 'images').exists():
                image_dir = str(Path(dataset_path) / 'images')
                label_dir = str(Path(dataset_path) / 'labels')
            else:
                image_dir = str(dataset_path)
                label_dir = str(Path(dataset_path).parent / 'labels')
        else:
            raise ValueError(f"Dataset path not found: {dataset_path}")
        
        print(f"  Images: {image_dir}")
        print(f"  Labels: {label_dir}")
    else:
        if not args.labels:
            raise ValueError("--labels is required when using --images")
        image_dir = args.images
        label_dir = args.labels
    
    test_ensemble_rigorous(
        model_paths=args.models,
        image_dir=image_dir,
        labels_dir=label_dir,
        weights=args.weights,
        method=args.method,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
