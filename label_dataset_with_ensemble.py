"""
Comprehensive Dataset Labeling Script with Ensemble Model
Supports 3 output modes:
1. Standard YOLO format (images + txt labels)
2. Images with visualized bounding boxes
3. Live video labeling with predictions

Usage:
    # Mode 1: Standard labels
    python label_dataset_with_ensemble.py --mode standard --input-dir ./unlabeled_images --output-dir ./labeled_dataset
    
    # Mode 2: Visualized bounding boxes
    python label_dataset_with_ensemble.py --mode visualized --input-dir ./images --output-dir ./visualized_output
    
    # Mode 3: Video labeling
    python label_dataset_with_ensemble.py --mode video --video-path ./underwater_footage.mp4 --output-path ./labeled_video.mp4
"""

import argparse
import cv2
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime


class EnsembleLabeler:
    """Ensemble model for high-quality dataset labeling"""
    
    def __init__(self, model_paths, weights=[0.40, 0.30, 0.15, 0.15], 
                 conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize ensemble labeler
        
        Args:
            model_paths: List of paths to model weights
            weights: Weight for each model in ensemble
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
        """
        self.models = [YOLO(path) for path in model_paths]
        self.weights = np.array(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Fish class names (update based on your dataset)
        self.class_names = {
            0: "Grunt Fish",
            1: "Parrot Fish", 
            2: "Surgeon Fish"
        }
        
        # Colors for visualization (BGR)
        self.class_colors = {
            0: (0, 255, 0),    # Green for Grunt
            1: (255, 0, 0),    # Blue for Parrot  
            2: (0, 165, 255)   # Orange for Surgeon
        }
    
    def predict_ensemble(self, image, use_tta=True):
        """
        Run ensemble prediction on a single image
        
        Args:
            image: Input image (numpy array or path)
            use_tta: Whether to use test-time augmentation
            
        Returns:
            boxes: List of [x1, y1, x2, y2, confidence, class_id]
        """
        all_predictions = []
        
        for i, model in enumerate(self.models):
            # Standard prediction
            results = model.predict(image, conf=self.conf_threshold, 
                                   iou=self.iou_threshold, verbose=False)[0]
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                # Weight the confidences
                weighted_confs = confs * self.weights[i]
                
                for box, conf, cls in zip(boxes, weighted_confs, classes):
                    all_predictions.append([
                        box[0], box[1], box[2], box[3], conf, int(cls)
                    ])
            
            # TTA: Horizontal flip
            if use_tta and isinstance(image, np.ndarray):
                flipped = cv2.flip(image, 1)
                results_flip = model.predict(flipped, conf=self.conf_threshold,
                                            iou=self.iou_threshold, verbose=False)[0]
                
                if results_flip.boxes is not None and len(results_flip.boxes) > 0:
                    boxes = results_flip.boxes.xyxy.cpu().numpy()
                    confs = results_flip.boxes.conf.cpu().numpy()
                    classes = results_flip.boxes.cls.cpu().numpy()
                    
                    # Flip boxes back
                    img_width = image.shape[1]
                    boxes[:, [0, 2]] = img_width - boxes[:, [2, 0]]
                    
                    weighted_confs = confs * self.weights[i] * 0.5  # Lower weight for TTA
                    
                    for box, conf, cls in zip(boxes, weighted_confs, classes):
                        all_predictions.append([
                            box[0], box[1], box[2], box[3], conf, int(cls)
                        ])
        
        # Apply NMS to ensemble predictions
        if len(all_predictions) == 0:
            return []
        
        predictions = np.array(all_predictions)
        final_boxes = self.non_max_suppression(predictions)
        
        return final_boxes
    
    def non_max_suppression(self, predictions):
        """Apply NMS to ensemble predictions"""
        if len(predictions) == 0:
            return []
        
        # Group by class
        unique_classes = np.unique(predictions[:, 5])
        final_boxes = []
        
        for cls in unique_classes:
            cls_predictions = predictions[predictions[:, 5] == cls]
            
            # Sort by confidence
            indices = np.argsort(cls_predictions[:, 4])[::-1]
            cls_predictions = cls_predictions[indices]
            
            keep = []
            while len(cls_predictions) > 0:
                keep.append(cls_predictions[0])
                
                if len(cls_predictions) == 1:
                    break
                
                # Calculate IoU with remaining boxes
                ious = self.calculate_iou(cls_predictions[0, :4], 
                                         cls_predictions[1:, :4])
                
                # Keep boxes with IoU below threshold
                cls_predictions = cls_predictions[1:][ious < self.iou_threshold]
            
            final_boxes.extend(keep)
        
        return final_boxes
    
    @staticmethod
    def calculate_iou(box1, boxes):
        """Calculate IoU between one box and multiple boxes"""
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = box1_area + boxes_area - intersection
        
        return intersection / (union + 1e-6)


def label_standard_format(labeler, input_dir, output_dir):
    """
    Mode 1: Generate standard YOLO format labels
    
    Output structure:
        output_dir/
            images/
                image1.jpg
                image2.jpg
            labels/
                image1.txt
                image2.txt
    """
    print("\n=== MODE 1: Standard YOLO Format Labeling ===")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images to label")
    
    # Statistics
    total_detections = 0
    detections_per_class = {0: 0, 1: 0, 2: 0}
    
    # Process each image
    for img_path in tqdm(image_files, desc="Labeling images"):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        height, width = image.shape[:2]
        
        # Get predictions
        predictions = labeler.predict_ensemble(image, use_tta=True)
        
        # Copy image
        output_img_path = images_dir / img_path.name
        shutil.copy2(img_path, output_img_path)
        
        # Write labels in YOLO format
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        with open(label_path, 'w') as f:
            for pred in predictions:
                x1, y1, x2, y2, conf, cls = pred
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                # Write: class x_center y_center width height
                f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                
                total_detections += 1
                detections_per_class[int(cls)] += 1
    
    # Write dataset.yaml
    yaml_content = f"""# Underwater Fish Dataset - Auto-labeled with Ensemble Model
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {output_path.absolute()}
train: images
val: images  # Update paths as needed
test: images

nc: 3  # number of classes
names: {list(labeler.class_names.values())}

# Auto-labeling Statistics:
# Total images: {len(image_files)}
# Total detections: {total_detections}
# Grunt Fish: {detections_per_class[0]}
# Parrot Fish: {detections_per_class[1]}
# Surgeon Fish: {detections_per_class[2]}
"""
    
    with open(output_path / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    # Save statistics
    stats = {
        "total_images": len(image_files),
        "total_detections": total_detections,
        "detections_per_class": detections_per_class,
        "class_names": labeler.class_names,
        "generation_date": datetime.now().isoformat()
    }
    
    with open(output_path / "labeling_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Labeling complete!")
    print(f"   Images: {images_dir}")
    print(f"   Labels: {labels_dir}")
    print(f"   Total detections: {total_detections}")
    print(f"   - Grunt Fish: {detections_per_class[0]}")
    print(f"   - Parrot Fish: {detections_per_class[1]}")
    print(f"   - Surgeon Fish: {detections_per_class[2]}")


def label_with_visualization(labeler, input_dir, output_dir, show_confidence=True):
    """
    Mode 2: Generate images with visualized bounding boxes
    
    Output: Images with drawn bounding boxes, labels, and confidence scores
    """
    print("\n=== MODE 2: Visualized Bounding Boxes ===")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images to visualize")
    
    for img_path in tqdm(image_files, desc="Creating visualizations"):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Get predictions
        predictions = labeler.predict_ensemble(image, use_tta=True)
        
        # Draw predictions
        for pred in predictions:
            x1, y1, x2, y2, conf, cls = pred
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)
            
            # Get color and label
            color = labeler.class_colors[cls]
            label = labeler.class_names[cls]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            if show_confidence:
                text = f"{label} {conf:.2f}"
            else:
                text = label
            
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(image, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(image, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save visualized image
        output_img_path = output_path / f"{img_path.stem}_labeled{img_path.suffix}"
        cv2.imwrite(str(output_img_path), image)
    
    print(f"\n✅ Visualization complete! Output: {output_path}")


def label_video_live(labeler, video_path, output_path, fps=None, show_fps=True):
    """
    Mode 3: Label video with live predictions
    
    Args:
        video_path: Input video file
        output_path: Output video file
        fps: Output FPS (None = same as input)
        show_fps: Show FPS counter on video
    """
    print("\n=== MODE 3: Live Video Labeling ===")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_fps = fps if fps else input_fps
    
    print(f"Video: {width}x{height} @ {input_fps} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Output FPS: {output_fps}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    frame_count = 0
    detections_count = {0: 0, 1: 0, 2: 0}
    
    # Process video
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get predictions
        predictions = labeler.predict_ensemble(frame, use_tta=False)  # Faster without TTA
        
        # Draw predictions
        for pred in predictions:
            x1, y1, x2, y2, conf, cls = pred
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)
            
            detections_count[cls] += 1
            
            # Get color and label
            color = labeler.class_colors[cls]
            label = labeler.class_names[cls]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"{label} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(frame, (x1, y1 - text_height - 10),
                         (x1 + text_width, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame info
        if show_fps:
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(predictions)}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✅ Video labeling complete!")
    print(f"   Output: {output_path}")
    print(f"   Frames processed: {frame_count}")
    print(f"   Total detections:")
    print(f"   - Grunt Fish: {detections_count[0]}")
    print(f"   - Parrot Fish: {detections_count[1]}")
    print(f"   - Surgeon Fish: {detections_count[2]}")


def main():
    parser = argparse.ArgumentParser(description="Label dataset with ensemble model")
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['standard', 'visualized', 'video'],
                       help='Labeling mode')
    
    # Model configuration
    parser.add_argument('--models', type=str, nargs='+',
                       default=[
                           'runs/detect/extreme_stable_v1/weights/best.pt',
                           'runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt',
                           'runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt',
                           'runs/detect/fish_s_s_cosine_ultra_v1/weights/best.pt'
                       ],
                       help='Model weights paths')
    parser.add_argument('--weights', type=float, nargs='+',
                       default=[0.40, 0.30, 0.15, 0.15],
                       help='Ensemble weights')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold')
    
    # Input/Output paths
    parser.add_argument('--input-dir', type=str, help='Input directory for images')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--video-path', type=str, help='Input video path (for video mode)')
    parser.add_argument('--output-path', type=str, help='Output video path (for video mode)')
    
    # Additional options
    parser.add_argument('--no-tta', action='store_true',
                       help='Disable test-time augmentation')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Hide confidence scores in visualization')
    parser.add_argument('--fps', type=int, default=None,
                       help='Output video FPS')
    
    args = parser.parse_args()
    
    # Initialize ensemble labeler
    print("Loading ensemble models...")
    labeler = EnsembleLabeler(
        model_paths=args.models,
        weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    print("✓ Models loaded successfully")
    
    # Run appropriate mode
    if args.mode == 'standard':
        if not args.input_dir or not args.output_dir:
            print("Error: --input-dir and --output-dir required for standard mode")
            return
        label_standard_format(labeler, args.input_dir, args.output_dir)
    
    elif args.mode == 'visualized':
        if not args.input_dir or not args.output_dir:
            print("Error: --input-dir and --output-dir required for visualized mode")
            return
        label_with_visualization(labeler, args.input_dir, args.output_dir,
                                show_confidence=not args.no_confidence)
    
    elif args.mode == 'video':
        if not args.video_path or not args.output_path:
            print("Error: --video-path and --output-path required for video mode")
            return
        label_video_live(labeler, args.video_path, args.output_path, 
                        fps=args.fps, show_fps=True)


if __name__ == "__main__":
    main()
