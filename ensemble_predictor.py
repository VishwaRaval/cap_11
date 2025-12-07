#!/usr/bin/env python3
"""
Ensemble Prediction Script for Fish Detection
Combines multiple YOLOv11 models for improved accuracy
"""

import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
from collections import defaultdict
import json
from typing import List, Dict, Tuple
import time

class FishEnsemble:
    """Ensemble predictor combining multiple YOLO models"""
    
    def __init__(self, model_paths: List[str], weights: List[float] = None, 
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize ensemble with multiple models
        
        Args:
            model_paths: List of paths to .pt model files
            weights: Optional weights for each model (must sum to 1.0)
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
        """
        self.models = []
        self.model_names = []
        
        print("Loading models...")
        for path in model_paths:
            model = YOLO(path)
            self.models.append(model)
            self.model_names.append(Path(path).parent.parent.name)
            print(f"  ✓ Loaded: {Path(path).parent.parent.name}")
        
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            assert len(weights) == len(self.models), "Weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
            self.weights = weights
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"\nEnsemble Configuration:")
        print(f"  Models: {len(self.models)}")
        print(f"  Weights: {self.weights}")
        print(f"  Confidence: {conf_threshold}")
        print(f"  IoU: {iou_threshold}")
    
    def predict_single(self, image_path: str, model_idx: int) -> Dict:
        """Get predictions from a single model"""
        results = self.models[model_idx].predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        
        return {
            'boxes': boxes,
            'confidences': confs,
            'classes': classes
        }
    
    def weighted_boxes_fusion(self, all_predictions: List[Dict]) -> Dict:
        """
        Weighted Boxes Fusion (WBF) ensemble method
        Combines overlapping boxes from different models
        """
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for pred, weight in zip(all_predictions, self.weights):
            if len(pred['boxes']) > 0:
                all_boxes.append(pred['boxes'])
                all_scores.append(pred['confidences'] * weight)
                all_labels.append(pred['classes'].astype(int))
        
        if not all_boxes:
            return {'boxes': np.array([]), 'confidences': np.array([]), 'classes': np.array([])}
        
        # Simple NMS-based fusion
        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        
        # Apply NMS
        keep_indices = self.nms(boxes, scores, self.iou_threshold)
        
        return {
            'boxes': boxes[keep_indices],
            'confidences': scores[keep_indices],
            'classes': labels[keep_indices]
        }
    
    def majority_voting(self, all_predictions: List[Dict]) -> Dict:
        """
        Majority voting ensemble - only keep boxes that appear in multiple models
        """
        # Group boxes by class and spatial location
        class_boxes = defaultdict(list)
        
        for pred, weight in zip(all_predictions, self.weights):
            for box, conf, cls in zip(pred['boxes'], pred['confidences'], pred['classes']):
                class_boxes[int(cls)].append({
                    'box': box,
                    'conf': conf * weight,
                    'model_weight': weight
                })
        
        # For each class, cluster nearby boxes
        final_boxes = []
        final_confs = []
        final_classes = []
        
        for cls, boxes_list in class_boxes.items():
            clusters = self.cluster_boxes(boxes_list)
            
            for cluster in clusters:
                if len(cluster) >= 2:  # At least 2 models agree
                    # Average box coordinates weighted by confidence
                    total_weight = sum(b['conf'] for b in cluster)
                    avg_box = np.average(
                        [b['box'] for b in cluster],
                        axis=0,
                        weights=[b['conf'] for b in cluster]
                    )
                    avg_conf = total_weight / len(cluster)
                    
                    final_boxes.append(avg_box)
                    final_confs.append(avg_conf)
                    final_classes.append(cls)
        
        if not final_boxes:
            return {'boxes': np.array([]), 'confidences': np.array([]), 'classes': np.array([])}
        
        return {
            'boxes': np.array(final_boxes),
            'confidences': np.array(final_confs),
            'classes': np.array(final_classes)
        }
    
    def cluster_boxes(self, boxes_list: List[Dict], iou_threshold: float = 0.5) -> List[List[Dict]]:
        """Cluster overlapping boxes"""
        if not boxes_list:
            return []
        
        clusters = []
        used = set()
        
        for i, box1 in enumerate(boxes_list):
            if i in used:
                continue
            
            cluster = [box1]
            used.add(i)
            
            for j, box2 in enumerate(boxes_list):
                if j <= i or j in used:
                    continue
                
                iou = self.calculate_iou(box1['box'], box2['box'])
                if iou > iou_threshold:
                    cluster.append(box2)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
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
    
    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def predict(self, image_path: str, method: str = 'wbf') -> Dict:
        """
        Run ensemble prediction
        
        Args:
            image_path: Path to image
            method: 'wbf' (weighted boxes fusion) or 'voting' (majority voting)
        
        Returns:
            Dictionary with boxes, confidences, and classes
        """
        # Get predictions from all models
        all_predictions = []
        for i in range(len(self.models)):
            pred = self.predict_single(image_path, i)
            all_predictions.append(pred)
        
        # Combine predictions
        if method == 'wbf':
            return self.weighted_boxes_fusion(all_predictions)
        elif method == 'voting':
            return self.majority_voting(all_predictions)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def evaluate_dataset(self, image_dir: str, labels_dir: str, method: str = 'wbf') -> Dict:
        """
        Evaluate ensemble on a dataset
        
        Args:
            image_dir: Directory with images
            labels_dir: Directory with YOLO format labels
            method: Ensemble method
        
        Returns:
            Dictionary with evaluation metrics
        """
        from tqdm import tqdm
        
        image_dir = Path(image_dir)
        labels_dir = Path(labels_dir)
        
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        all_true = []
        all_pred = []
        
        print(f"\nEvaluating ensemble on {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="Processing"):
            # Get ground truth
            label_path = labels_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                continue
            
            with open(label_path, 'r') as f:
                gt_classes = [int(line.split()[0]) for line in f.readlines()]
            
            # Get predictions
            pred = self.predict(str(img_path), method=method)
            pred_classes = pred['classes'].astype(int).tolist()
            
            all_true.extend(gt_classes)
            all_pred.extend(pred_classes)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(all_true, all_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(all_true, all_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'num_samples': len(all_true)
        }


def test_ensemble_combinations():
    """Test different ensemble combinations"""
    
    # Define top models (update paths as needed)
    top_models = {
        'extreme_stable_v1': 'runs/detect/extreme_stable_v1/weights/best.pt',
        'best_cosine_finetune': 'runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt',
        'large_precision': 'runs/detect/large_precision_v1_scratch/weights/best.pt',
        'extreme_stable_v2': 'runs/detect/extreme_stable_v2_native/weights/best.pt',
        'moderate_balanced': 'runs/detect/moderate_balanced_v1/weights/best.pt',
    }
    
    # Test different combinations
    combinations = [
        # Top 3 diverse models
        {
            'name': 'Top3_Diverse',
            'models': ['extreme_stable_v1', 'best_cosine_finetune', 'large_precision'],
            'weights': [0.4, 0.35, 0.25]  # Weight by accuracy
        },
        # Top 5 models
        {
            'name': 'Top5_Diverse',
            'models': ['extreme_stable_v1', 'best_cosine_finetune', 'large_precision', 
                      'extreme_stable_v2', 'moderate_balanced'],
            'weights': [0.25, 0.25, 0.20, 0.15, 0.15]
        },
        # Precision-Recall balance
        {
            'name': 'PrecRecall_Balance',
            'models': ['extreme_stable_v1', 'extreme_stable_v2', 'moderate_balanced'],
            'weights': [0.5, 0.3, 0.2]
        },
    ]
    
    results = []
    
    for combo in combinations:
        print(f"\n{'='*80}")
        print(f"Testing: {combo['name']}")
        print(f"{'='*80}")
        
        model_paths = [top_models[m] for m in combo['models']]
        
        ensemble = FishEnsemble(
            model_paths=model_paths,
            weights=combo['weights'],
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        # Test both methods
        for method in ['wbf', 'voting']:
            print(f"\nMethod: {method.upper()}")
            
            # Evaluate (update with your dataset path)
            metrics = ensemble.evaluate_dataset(
                image_dir='path/to/test/images',
                labels_dir='path/to/test/labels',
                method=method
            )
            
            result = {
                'combination': combo['name'],
                'method': method,
                'models': combo['models'],
                'weights': combo['weights'],
                **metrics
            }
            results.append(result)
            
            print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"  Precision: {metrics['precision']*100:.2f}%")
            print(f"  Recall: {metrics['recall']*100:.2f}%")
            print(f"  F1: {metrics['f1']*100:.2f}%")
    
    # Save results
    with open('ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best combination
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n{'='*80}")
    print(f"🏆 BEST ENSEMBLE")
    print(f"{'='*80}")
    print(f"Combination: {best['combination']}")
    print(f"Method: {best['method']}")
    print(f"Accuracy: {best['accuracy']*100:.2f}%")
    print(f"Models: {best['models']}")
    print(f"Weights: {best['weights']}")
    
    return results


if __name__ == "__main__":
    # Example usage
    test_ensemble_combinations()
