#!/usr/bin/env python3
"""
TRUE Ensemble Inference - Box-Level Fusion
This actually combines predictions from multiple models, not just averages metrics

Strategy:
1. Load all 4 models
2. Run inference on each image
3. Combine all predicted boxes using Weighted Boxes Fusion (WBF)
4. Evaluate combined predictions against ground truth
5. Calculate REAL ensemble accuracy

This will give us the actual ensemble performance, not an estimate.
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import yaml
from collections import defaultdict

# Weighted Boxes Fusion implementation
def bbox_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights, iou_thr=0.5, conf_type='avg'):
    """
    Weighted Boxes Fusion
    
    Args:
        boxes_list: List of [N, 4] arrays (one per model)
        scores_list: List of [N] arrays (one per model)
        labels_list: List of [N] arrays (one per model)
        weights: List of weights for each model
        iou_thr: IoU threshold for fusion
        conf_type: 'avg', 'max', or 'weighted_avg'
    
    Returns:
        fused_boxes, fused_scores, fused_labels
    """
    if not boxes_list:
        return np.array([]), np.array([]), np.array([])
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Collect all boxes with their metadata
    all_boxes = []
    for model_idx, (boxes, scores, labels, weight) in enumerate(
        zip(boxes_list, scores_list, labels_list, weights)
    ):
        for box, score, label in zip(boxes, scores, labels):
            all_boxes.append({
                'box': box,
                'score': score,
                'label': int(label),
                'model_idx': model_idx,
                'weight': weight
            })
    
    if not all_boxes:
        return np.array([]), np.array([]), np.array([])
    
    # Sort by score descending
    all_boxes.sort(key=lambda x: x['score'], reverse=True)
    
    # Fusion algorithm
    fused = []
    used = [False] * len(all_boxes)
    
    for i, box_i in enumerate(all_boxes):
        if used[i]:
            continue
        
        # Find all overlapping boxes with same class
        cluster = [box_i]
        used[i] = True
        
        for j, box_j in enumerate(all_boxes[i+1:], start=i+1):
            if used[j]:
                continue
            if box_j['label'] != box_i['label']:
                continue
            
            iou = bbox_iou(box_i['box'], box_j['box'])
            if iou >= iou_thr:
                cluster.append(box_j)
                used[j] = True
        
        # Fuse cluster
        if conf_type == 'max':
            fused_score = max(b['score'] for b in cluster)
        elif conf_type == 'weighted_avg':
            total_weight = sum(b['weight'] for b in cluster)
            fused_score = sum(b['score'] * b['weight'] for b in cluster) / total_weight
        else:  # 'avg'
            fused_score = sum(b['score'] for b in cluster) / len(cluster)
        
        # Weighted average of boxes
        total_weight = sum(b['weight'] * b['score'] for b in cluster)
        fused_box = np.sum([
            b['box'] * b['weight'] * b['score'] for b in cluster
        ], axis=0) / total_weight
        
        fused.append({
            'box': fused_box,
            'score': fused_score,
            'label': box_i['label'],
            'n_models': len(set(b['model_idx'] for b in cluster))
        })
    
    if not fused:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to arrays
    fused_boxes = np.array([f['box'] for f in fused])
    fused_scores = np.array([f['score'] for f in fused])
    fused_labels = np.array([f['label'] for f in fused])
    
    return fused_boxes, fused_scores, fused_labels


def evaluate_predictions(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Evaluate predictions against ground truth
    Returns: TP, FP, FN counts per class
    """
    n_classes = 3
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    
    # Match predictions to ground truth
    matched_gt = set()
    
    for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
        pred_label = int(pred_label)
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_idx in matched_gt:
                continue
            if int(gt_label) != pred_label:
                continue
            
            iou = bbox_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[pred_label] += 1
            matched_gt.add(best_gt_idx)
        else:
            fp[pred_label] += 1
    
    # Count false negatives (unmatched ground truth)
    for gt_idx, gt_label in enumerate(gt_labels):
        if gt_idx not in matched_gt:
            fn[int(gt_label)] += 1
    
    return tp, fp, fn


def load_ground_truth(label_path):
    """Load ground truth from YOLO format label file"""
    if not label_path.exists():
        return np.array([]), np.array([])
    
    boxes = []
    labels = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            label = int(parts[0])
            # Convert from YOLO format (x_center, y_center, w, h) to (x1, y1, x2, y2)
            x_center, y_center, w, h = map(float, parts[1:5])
            x1 = x_center - w/2
            y1 = y_center - h/2
            x2 = x_center + w/2
            y2 = y_center + h/2
            
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
    
    return np.array(boxes), np.array(labels)


def true_ensemble_validation(model_paths, model_weights, data_yaml, conf_threshold=0.25, 
                             iou_fusion=0.5, iou_eval=0.5):
    """
    Run TRUE ensemble validation with box-level fusion
    
    Returns:
        Dictionary with precision, recall, f1, avg_accuracy per class and overall
    """
    print('\n' + '='*80)
    print('TRUE ENSEMBLE VALIDATION - BOX-LEVEL FUSION')
    print('='*80)
    print(f'\nLoading {len(model_paths)} models...')
    
    # Load models
    models = []
    for i, path in enumerate(model_paths, 1):
        print(f'  [{i}/{len(model_paths)}] Loading {Path(path).parent.parent.name}...', end=' ')
        models.append(YOLO(path))
        print('✓')
    
    # Load dataset info
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_root = Path(data_config['path'])
    val_images = dataset_root / 'valid' / 'images'
    val_labels = dataset_root / 'valid' / 'labels'
    
    image_files = list(val_images.glob('*.jpg')) + list(val_images.glob('*.png'))
    
    print(f'\n✓ Found {len(image_files)} validation images')
    print(f'\nEnsemble settings:')
    print(f'  Confidence threshold: {conf_threshold}')
    print(f'  IoU fusion threshold: {iou_fusion}')
    print(f'  IoU eval threshold: {iou_eval}')
    print(f'  Model weights: {model_weights}')
    
    # Tracking metrics
    class_names = data_config.get('names', ['Grunt Fish', 'Parrot Fish', 'Surgeon Fish'])
    n_classes = len(class_names)
    
    total_tp = np.zeros(n_classes)
    total_fp = np.zeros(n_classes)
    total_fn = np.zeros(n_classes)
    
    print(f'\n' + '='*80)
    print('RUNNING ENSEMBLE INFERENCE ON VALIDATION SET')
    print('='*80)
    print()
    
    # Process each image
    for img_path in tqdm(image_files, desc='Ensemble inference'):
        # Get predictions from each model
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for model in models:
            results = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                iou=0.5,
                verbose=False
            )
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                result = results[0]
                # Convert to normalized coordinates [0, 1]
                boxes = result.boxes.xyxyn.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy()
                
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
            else:
                boxes_list.append(np.array([]))
                scores_list.append(np.array([]))
                labels_list.append(np.array([]))
        
        # Fuse predictions
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            weights=model_weights,
            iou_thr=iou_fusion,
            conf_type='weighted_avg'
        )
        
        # Load ground truth
        label_path = val_labels / (img_path.stem + '.txt')
        gt_boxes, gt_labels = load_ground_truth(label_path)
        
        # Evaluate
        tp, fp, fn = evaluate_predictions(
            fused_boxes, fused_labels, fused_scores,
            gt_boxes, gt_labels,
            iou_threshold=iou_eval
        )
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate metrics
    print('\n' + '='*80)
    print('TRUE ENSEMBLE RESULTS')
    print('='*80)
    
    results = {}
    
    # Per-class metrics
    print(f'\nPer-Class Metrics:')
    print('-'*80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print('-'*80)
    
    class_precisions = []
    class_recalls = []
    class_f1s = []
    
    for cls_idx, cls_name in enumerate(class_names):
        tp = total_tp[cls_idx]
        fp = total_fp[cls_idx]
        fn = total_fn[cls_idx]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)
        
        print(f"{cls_name:<20} {precision*100:>10.2f}% {recall*100:>10.2f}% {f1*100:>10.2f}%")
    
    # Overall metrics (mean of per-class)
    overall_precision = np.mean(class_precisions)
    overall_recall = np.mean(class_recalls)
    overall_f1 = np.mean(class_f1s)
    avg_accuracy = (overall_precision + overall_recall) / 2
    
    print('-'*80)
    print(f"{'Overall (mean)':<20} {overall_precision*100:>10.2f}% {overall_recall*100:>10.2f}% {overall_f1*100:>10.2f}%")
    print('-'*80)
    
    print(f'\n🎯 FINAL METRICS:')
    print(f'  Precision:     {overall_precision*100:.2f}%')
    print(f'  Recall:        {overall_recall*100:.2f}%')
    print(f'  F1 Score:      {overall_f1*100:.2f}%')
    print(f'  Avg Accuracy:  {avg_accuracy*100:.2f}%')
    
    print(f'\nTarget: 70.00%')
    if avg_accuracy >= 0.70:
        print(f'Status: ✅ TARGET MET!')
    else:
        print(f'Status: ❌ Gap: {(0.70-avg_accuracy)*100:.2f}%')
    
    print('='*80)
    
    results['precision'] = overall_precision
    results['recall'] = overall_recall
    results['f1'] = overall_f1
    results['avg_accuracy'] = avg_accuracy
    results['class_metrics'] = {
        'precisions': class_precisions,
        'recalls': class_recalls,
        'f1s': class_f1s
    }
    
    return results


def main():
    """Run true ensemble validation"""
    
    # Model paths and weights (based on individual performance)
    models_info = [
        {
            'path': 'runs/detect/extreme_stable_v1/weights/best.pt',
            'weight': 0.25,  # Best performer
            'name': 'extreme_stable_v1'
        },
        {
            'path': 'runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt',
            'weight': 0.25,
            'name': 'cosine_finetune'
        },
        {
            'path': 'runs/detect/fish_s_s_cosine_ultra_v1/weights/best.pt',
            'weight': 0.25,
            'name': 'cosine_ultra'
        },
        {
            'path': 'runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt',
            'weight': 0.25,
            'name': 'recall_optimized'
        },
    ]
    
    model_paths = [m['path'] for m in models_info]
    model_weights = [m['weight'] for m in models_info]
    
    # Check if models exist
    missing = []
    for info in models_info:
        if not Path(info['path']).exists():
            missing.append(info['name'])
    
    if missing:
        print(f"❌ Missing models: {', '.join(missing)}")
        print("\nTrying with only extreme_stable_v1...")
        model_paths = ['runs/detect/extreme_stable_v1/weights/best.pt']
        model_weights = [1.0]
    
    # Data YAML
    data_yaml = 'dataset_root/data.yaml'
    if not Path(data_yaml).exists():
        print(f"❌ data.yaml not found at: {data_yaml}")
        return
    
    print('\n' + '='*80)
    print('TRUE ENSEMBLE VALIDATION')
    print('='*80)
    print(f'\nModels to ensemble: {len(model_paths)}')
    for i, info in enumerate([m for m in models_info if Path(m['path']).exists()], 1):
        print(f"  {i}. {info['name']:<30} Weight: {info['weight']:.2f}")
    
    # Run validation with different settings
    print('\n' + '='*80)
    print('TESTING DIFFERENT ENSEMBLE CONFIGURATIONS')
    print('='*80)
    
    configs = [
        {'conf': 0.20, 'iou_fusion': 0.5, 'iou_eval': 0.5},
        {'conf': 0.25, 'iou_fusion': 0.5, 'iou_eval': 0.5},
        {'conf': 0.25, 'iou_fusion': 0.4, 'iou_eval': 0.5},
    ]
    
    best_acc = 0
    best_config = None
    best_results = None
    
    for config in configs:
        print(f'\n{"="*80}')
        print(f'Config: conf={config["conf"]:.2f}, fusion_iou={config["iou_fusion"]:.2f}')
        print(f'{"="*80}')
        
        results = true_ensemble_validation(
            model_paths, model_weights, data_yaml,
            conf_threshold=config['conf'],
            iou_fusion=config['iou_fusion'],
            iou_eval=config['iou_eval']
        )
        
        if results['avg_accuracy'] > best_acc:
            best_acc = results['avg_accuracy']
            best_config = config
            best_results = results
    
    # Final summary
    print('\n' + '='*80)
    print('🏆 BEST ENSEMBLE CONFIGURATION')
    print('='*80)
    print(f'\nConfiguration:')
    print(f"  Confidence: {best_config['conf']:.2f}")
    print(f"  Fusion IoU: {best_config['iou_fusion']:.2f}")
    print(f"  Eval IoU:   {best_config['iou_eval']:.2f}")
    print(f'\nResults:')
    print(f"  Precision:    {best_results['precision']*100:.2f}%")
    print(f"  Recall:       {best_results['recall']*100:.2f}%")
    print(f"  F1 Score:     {best_results['f1']*100:.2f}%")
    print(f"  Avg Accuracy: {best_results['avg_accuracy']*100:.2f}%")
    
    print(f'\n🎯 Target: 70.00%')
    if best_acc >= 0.70:
        print(f'   ✅ TARGET MET WITH TRUE ENSEMBLE!')
    else:
        gap = (0.70 - best_acc) * 100
        print(f'   ❌ Gap: {gap:.2f}%')
        
        if gap <= 2:
            print('\n💡 VERY CLOSE! Try adding:')
            print('   - Test-Time Augmentation (TTA)')
            print('   - Expected gain: +1-2%')
        elif gap <= 4:
            print('\n💡 Options to close gap:')
            print('   - Add TTA (+1-2%)')
            print('   - Tune confidence threshold')
            print('   - Train one more diverse model')
        else:
            print('\n💡 Significant gap remaining:')
            print('   - This is the real ensemble limit')
            print('   - Need better base models or different approach')
    
    print('='*80)
    
    # Save results
    with open('true_ensemble_results.txt', 'w') as f:
        f.write('TRUE ENSEMBLE RESULTS\n')
        f.write('='*80 + '\n\n')
        f.write(f"Best configuration:\n")
        f.write(f"  Confidence: {best_config['conf']:.2f}\n")
        f.write(f"  Fusion IoU: {best_config['iou_fusion']:.2f}\n")
        f.write(f"  Eval IoU:   {best_config['iou_eval']:.2f}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Precision:    {best_results['precision']*100:.2f}%\n")
        f.write(f"  Recall:       {best_results['recall']*100:.2f}%\n")
        f.write(f"  F1 Score:     {best_results['f1']*100:.2f}%\n")
        f.write(f"  Avg Accuracy: {best_results['avg_accuracy']*100:.2f}%\n")
    
    print(f'\n✓ Results saved to: true_ensemble_results.txt')


if __name__ == '__main__':
    main()
