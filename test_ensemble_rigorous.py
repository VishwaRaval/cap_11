#!/usr/bin/env python3
"""
Test ensemble with rigorous IoU-based evaluation
This is the REAL test of your 88.64% claim
"""

import sys
sys.path.append('.')

from ensemble_predictor import FishEnsemble
from rigorous_eval import calculate_iou, yolo_to_xyxy
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import cv2

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
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    # Track metrics
    class_tp = {0: 0, 1: 0, 2: 0}
    class_fp = {0: 0, 1: 0, 2: 0}
    class_fn = {0: 0, 1: 0, 2: 0}
    class_gt_count = {0: 0, 1: 0, 2: 0}
    
    print(f"\nProcessing {len(image_files)} images...")
    
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
    
    # Calculate metrics
    class_names = ['Grunt Fish', 'Parrot Fish', 'Surgeon Fish']
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")
    
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    total_gt = sum(class_gt_count.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"Overall Metrics (IoU >= {iou_threshold}):")
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--weights', nargs='+', type=float, default=None)
    parser.add_argument('--method', choices=['wbf', 'voting'], default='wbf')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.5)
    
    args = parser.parse_args()
    
    test_ensemble_rigorous(
        model_paths=args.models,
        image_dir=args.images,
        labels_dir=args.labels,
        weights=args.weights,
        method=args.method,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
