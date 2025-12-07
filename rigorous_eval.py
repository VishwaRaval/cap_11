#!/usr/bin/env python3
"""
Rigorous Ensemble Evaluation with IoU-based matching
Uses proper object detection metrics (mAP, IoU threshold)
"""

import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse


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


def evaluate_single_model(model_path: str, image_dir: str, labels_dir: str, 
                         conf_threshold: float = 0.25, iou_threshold: float = 0.5):
    """
    Rigorously evaluate a single model with IoU-based matching
    
    Args:
        model_path: Path to model .pt file
        image_dir: Directory with test images
        labels_dir: Directory with ground truth labels
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching (0.5 is standard)
    
    Returns:
        Dictionary with detailed metrics
    """
    model = YOLO(model_path)
    image_dir = Path(image_dir)
    labels_dir = Path(labels_dir)
    
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    # Track per-class metrics
    class_tp = {0: 0, 1: 0, 2: 0}
    class_fp = {0: 0, 1: 0, 2: 0}
    class_fn = {0: 0, 1: 0, 2: 0}
    class_gt_count = {0: 0, 1: 0, 2: 0}
    
    total_images = 0
    
    print(f"\nEvaluating: {Path(model_path).parent.parent.name}")
    print(f"IoU threshold: {iou_threshold}")
    
    for img_path in tqdm(image_files, desc="Processing"):
        # Get ground truth
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        
        # Read image dimensions
        import cv2
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
        
        # Get predictions
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False
        )[0]
        
        pred_boxes = results.boxes.xyxy.cpu().numpy()
        pred_classes = results.boxes.cls.cpu().numpy().astype(int)
        pred_confs = results.boxes.conf.cpu().numpy()
        
        # Match predictions to ground truth using IoU
        gt_matched = set()
        
        for pred_idx in range(len(pred_boxes)):
            pred_box = pred_boxes[pred_idx]
            pred_cls = pred_classes[pred_idx]
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in gt_matched:
                    continue
                
                gt_box = gt_boxes[gt_idx]
                gt_cls = gt_classes[gt_idx]
                
                # Only match same class
                if pred_cls != gt_cls:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= iou_threshold:
                # True positive
                class_tp[pred_cls] += 1
                gt_matched.add(best_gt_idx)
            else:
                # False positive
                class_fp[pred_cls] += 1
        
        # Count false negatives (unmatched ground truth)
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in gt_matched:
                class_fn[gt_classes[gt_idx]] += 1
        
        total_images += 1
    
    # Calculate metrics
    class_names = ['Grunt Fish', 'Parrot Fish', 'Surgeon Fish']
    
    results = {
        'model': Path(model_path).parent.parent.name,
        'iou_threshold': iou_threshold,
        'total_images': total_images,
        'per_class': {}
    }
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for cls in [0, 1, 2]:
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results['per_class'][class_names[cls]] = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'ground_truth': int(class_gt_count[cls]),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    results['overall'] = {
        'tp': int(total_tp),
        'fp': int(total_fp),
        'fn': int(total_fn),
        'precision': float(overall_precision),
        'recall': float(overall_recall),
        'f1': float(overall_f1)
    }
    
    return results


def compare_models(model_paths: List[str], image_dir: str, labels_dir: str,
                  conf_threshold: float = 0.25, iou_threshold: float = 0.5):
    """
    Compare multiple models side-by-side
    """
    all_results = []
    
    print(f"\n{'='*80}")
    print("RIGOROUS MODEL EVALUATION")
    print(f"{'='*80}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    
    for model_path in model_paths:
        results = evaluate_single_model(
            model_path, image_dir, labels_dir, conf_threshold, iou_threshold
        )
        all_results.append(results)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<40} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)
    
    for result in all_results:
        model_name = result['model'][:38]
        precision = result['overall']['precision'] * 100
        recall = result['overall']['recall'] * 100
        f1 = result['overall']['f1'] * 100
        
        print(f"{model_name:<40} {precision:>9.2f}% {recall:>9.2f}% {f1:>9.2f}%")
    
    print("\n" + "="*80)
    print("PER-CLASS BREAKDOWN")
    print("="*80 + "\n")
    
    for result in all_results:
        print(f"\n{result['model']}:")
        print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
        print("  " + "-" * 76)
        
        for class_name, metrics in result['per_class'].items():
            prec = metrics['precision'] * 100
            rec = metrics['recall'] * 100
            f1 = metrics['f1'] * 100
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            print(f"  {class_name:<15} {prec:>9.2f}% {rec:>9.2f}% {f1:>9.2f}% {tp:>6d} {fp:>6d} {fn:>6d}")
    
    # Save results
    output_file = 'rigorous_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Rigorous model evaluation with IoU-based matching')
    parser.add_argument('--models', nargs='+', required=True, help='Model .pt files')
    parser.add_argument('--images', required=True, help='Test images directory')
    parser.add_argument('--labels', required=True, help='Test labels directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for matching')
    
    args = parser.parse_args()
    
    compare_models(
        model_paths=args.models,
        image_dir=args.images,
        labels_dir=args.labels,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )


if __name__ == "__main__":
    main()
