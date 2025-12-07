#!/usr/bin/env python3
"""
Ensemble with Test-Time Augmentation (TTA)
Augment images during inference to improve robustness
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ensemble_predictor import FishEnsemble
import numpy as np
from tqdm import tqdm
import json
import cv2
import yaml
import argparse


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
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
    """Convert YOLO format to xyxy"""
    x_center, y_center, width, height = yolo_box
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    return [x1, y1, x2, y2]


def apply_tta_augmentations(image):
    """
    Apply test-time augmentations
    Returns list of (augmented_image, reverse_transform_function)
    """
    h, w = image.shape[:2]
    augmentations = []
    
    # Original
    augmentations.append((image.copy(), lambda boxes: boxes))
    
    # Horizontal flip
    flipped = cv2.flip(image, 1)
    def reverse_hflip(boxes):
        boxes_copy = boxes.copy()
        boxes_copy[:, [0, 2]] = w - boxes_copy[:, [2, 0]]
        return boxes_copy
    augmentations.append((flipped, reverse_hflip))
    
    # Brightness adjustments (underwater images)
    # Slightly brighter
    bright = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
    augmentations.append((bright, lambda boxes: boxes))
    
    # Slightly darker
    dark = cv2.convertScaleAbs(image, alpha=0.9, beta=-10)
    augmentations.append((dark, lambda boxes: boxes))
    
    return augmentations


def ensemble_with_tta(ensemble, image_path, method='wbf', enable_tta=True):
    """
    Run ensemble with test-time augmentation
    
    Args:
        ensemble: FishEnsemble object
        image_path: Path to image
        method: Ensemble method
        enable_tta: Whether to use TTA
    
    Returns:
        Combined predictions
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {'boxes': np.array([]), 'confidences': np.array([]), 'classes': np.array([])}
    
    if not enable_tta:
        # Just run normal ensemble
        return ensemble.predict(str(image_path), method=method)
    
    # Get augmentations
    augmentations = apply_tta_augmentations(img)
    
    all_boxes = []
    all_confs = []
    all_classes = []
    
    # Run ensemble on each augmentation
    for aug_img, reverse_fn in augmentations:
        # Save augmented image temporarily
        temp_path = '/tmp/tta_temp.jpg'
        cv2.imwrite(temp_path, aug_img)
        
        # Run ensemble
        pred = ensemble.predict(temp_path, method=method)
        
        if len(pred['boxes']) > 0:
            # Reverse transformation
            reversed_boxes = reverse_fn(pred['boxes'])
            
            all_boxes.append(reversed_boxes)
            all_confs.append(pred['confidences'])
            all_classes.append(pred['classes'])
    
    if not all_boxes:
        return {'boxes': np.array([]), 'confidences': np.array([]), 'classes': np.array([])}
    
    # Combine all predictions
    all_boxes = np.vstack(all_boxes)
    all_confs = np.concatenate(all_confs)
    all_classes = np.concatenate(all_classes)
    
    # Apply NMS to merged predictions
    keep_indices = ensemble.nms(all_boxes, all_confs, ensemble.iou_threshold)
    
    return {
        'boxes': all_boxes[keep_indices],
        'confidences': all_confs[keep_indices],
        'classes': all_classes[keep_indices]
    }


def evaluate_with_tta(model_paths, image_dir, labels_dir, 
                      weights=None, method='wbf', conf_threshold=0.25, 
                      iou_threshold=0.5, enable_tta=True):
    """
    Evaluate ensemble with TTA
    """
    print(f"\n{'='*80}")
    print("ENSEMBLE EVALUATION WITH TEST-TIME AUGMENTATION")
    print(f"{'='*80}")
    print(f"TTA Enabled: {enable_tta}")
    print(f"Method: {method}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    
    # Create ensemble
    ensemble = FishEnsemble(
        model_paths=model_paths,
        weights=weights,
        conf_threshold=conf_threshold,
        iou_threshold=0.45
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
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        
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
        
        # Get predictions with TTA
        pred = ensemble_with_tta(ensemble, str(img_path), method=method, enable_tta=enable_tta)
        pred_boxes = pred['boxes']
        pred_classes = pred['classes'].astype(int)
        
        # Match predictions to ground truth
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
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Overall Metrics (IoU >= {iou_threshold}):")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"\n  TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    
    print(f"\nPer-Class Metrics:")
    for cls, name in enumerate(class_names):
        tp, fp, fn = class_tp[cls], class_fp[cls], class_fn[cls]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"  {name:15s}: Prec={prec*100:5.1f}%  Rec={rec*100:5.1f}%")
    
    if f1 >= 0.70:
        print(f"\n🎉 SUCCESS! Achieved 70% F1 target!")
    else:
        print(f"\n📊 F1: {f1*100:.2f}% (need {(0.70-f1)*100:.2f}% more for 70%)")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data')
    data_group.add_argument('--images')
    
    parser.add_argument('--labels')
    parser.add_argument('--weights', nargs='+', type=float, default=None)
    parser.add_argument('--method', choices=['wbf', 'voting'], default='wbf')
    parser.add_argument('--conf', type=float, default=0.45)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    
    args = parser.parse_args()
    
    # Parse paths
    if args.data:
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        yaml_dir = Path(args.data).parent
        dataset_path = data_config.get('test') or data_config.get('val')
        if not Path(dataset_path).is_absolute():
            dataset_path = yaml_dir / dataset_path
        if (Path(dataset_path) / 'images').exists():
            image_dir = str(Path(dataset_path) / 'images')
            label_dir = str(Path(dataset_path) / 'labels')
        else:
            image_dir = str(dataset_path)
            label_dir = str(Path(dataset_path).parent / 'labels')
    else:
        image_dir = args.images
        label_dir = args.labels
    
    evaluate_with_tta(
        model_paths=args.models,
        image_dir=image_dir,
        labels_dir=label_dir,
        weights=args.weights,
        method=args.method,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        enable_tta=not args.no_tta
    )
