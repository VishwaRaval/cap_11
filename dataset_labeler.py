#!/usr/bin/env python3
"""
Dataset Labeling and Visualization Script
Creates different types of visualizations for fish detection models
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import time

class FishLabeler:
    """Label and visualize fish detection results"""
    
    # Class names and colors
    CLASS_NAMES = {
        0: "Grunt Fish",
        1: "Parrot Fish",
        2: "Surgeon Fish"
    }
    
    CLASS_COLORS = {
        0: (0, 255, 0),      # Green for Grunt
        1: (255, 0, 0),      # Blue for Parrot
        2: (0, 165, 255)     # Orange for Surgeon
    }
    
    def __init__(self, model_path: str, model_name: str = None):
        """
        Initialize labeler with a model
        
        Args:
            model_path: Path to .pt model file
            model_name: Optional name for the model
        """
        self.model = YOLO(model_path)
        self.model_name = model_name or Path(model_path).parent.parent.name
        print(f"✓ Loaded model: {self.model_name}")
    
    def create_labels_only(self, image_path: str, output_dir: str, 
                           conf_threshold: float = 0.25) -> Dict:
        """
        Create YOLO format labels file (text file with class and bbox)
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save labels
            conf_threshold: Confidence threshold
        
        Returns:
            Dictionary with detection statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )[0]
        
        # Get image dimensions
        img_height, img_width = results.orig_shape
        
        # Create label file
        label_path = output_dir / (Path(image_path).stem + '.txt')
        
        detections = []
        with open(label_path, 'w') as f:
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Convert to YOLO format (normalized xywh)
                xyxy = box.xyxy[0].cpu().numpy()
                x_center = ((xyxy[0] + xyxy[2]) / 2) / img_width
                y_center = ((xyxy[1] + xyxy[3]) / 2) / img_height
                width = (xyxy[2] - xyxy[0]) / img_width
                height = (xyxy[3] - xyxy[1]) / img_height
                
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                detections.append({
                    'class': cls,
                    'class_name': self.CLASS_NAMES[cls],
                    'confidence': conf,
                    'bbox': xyxy.tolist()
                })
        
        return {
            'image': Path(image_path).name,
            'num_detections': len(detections),
            'detections': detections,
            'label_file': str(label_path)
        }
    
    def create_labeled_image(self, image_path: str, output_dir: str,
                            conf_threshold: float = 0.25,
                            show_confidence: bool = True,
                            show_bbox: bool = True) -> str:
        """
        Create image with bounding boxes and labels overlaid
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save labeled images
            conf_threshold: Confidence threshold
            show_confidence: Whether to show confidence scores
            show_bbox: Whether to show bounding boxes
        
        Returns:
            Path to saved image
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )[0]
        
        # Draw detections
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            color = self.CLASS_COLORS.get(cls, (255, 255, 255))
            
            # Draw bounding box
            if show_bbox:
                cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), 
                            color, 2)
            
            # Create label text
            label = self.CLASS_NAMES[cls]
            if show_confidence:
                label += f" {conf:.2f}"
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                image,
                (xyxy[0], xyxy[1] - text_height - baseline - 5),
                (xyxy[0] + text_width, xyxy[1]),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (xyxy[0], xyxy[1] - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Add model name to image
        cv2.putText(
            image,
            f"Model: {self.model_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )
        
        # Save image
        output_path = output_dir / f"{Path(image_path).stem}_labeled.jpg"
        cv2.imwrite(str(output_path), image)
        
        return str(output_path)
    
    def create_comparison_grid(self, image_path: str, models: List[Tuple[str, str]],
                              output_path: str, conf_threshold: float = 0.25) -> str:
        """
        Create comparison grid showing predictions from multiple models
        
        Args:
            image_path: Path to input image
            models: List of (model_path, model_name) tuples
            output_path: Path to save comparison image
            conf_threshold: Confidence threshold
        
        Returns:
            Path to saved comparison image
        """
        images = []
        
        # Original image
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Add title to original
        titled_original = original.copy()
        cv2.putText(
            titled_original,
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )
        images.append(titled_original)
        
        # Get predictions from each model
        for model_path, model_name in models:
            temp_labeler = FishLabeler(model_path, model_name)
            
            # Create labeled image
            labeled = original.copy()
            results = temp_labeler.model.predict(
                source=image_path,
                conf=conf_threshold,
                verbose=False
            )[0]
            
            # Draw detections
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                color = self.CLASS_COLORS.get(cls, (255, 255, 255))
                
                cv2.rectangle(labeled, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), 
                            color, 2)
                
                label = f"{self.CLASS_NAMES[cls]} {conf:.2f}"
                cv2.putText(
                    labeled,
                    label,
                    (xyxy[0], xyxy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            
            # Add model name
            cv2.putText(
                labeled,
                model_name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
            
            images.append(labeled)
        
        # Create grid
        n = len(images)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        # Resize all images to same size
        target_height = 480
        target_width = int(original.shape[1] * (target_height / original.shape[0]))
        
        resized = [cv2.resize(img, (target_width, target_height)) for img in images]
        
        # Create grid
        grid_rows = []
        for i in range(rows):
            row_images = resized[i*cols:(i+1)*cols]
            # Pad with black images if needed
            while len(row_images) < cols:
                row_images.append(np.zeros_like(resized[0]))
            grid_rows.append(np.hstack(row_images))
        
        grid = np.vstack(grid_rows)
        
        # Save
        cv2.imwrite(output_path, grid)
        return output_path
    
    def process_dataset(self, image_dir: str, output_base_dir: str,
                       conf_threshold: float = 0.25,
                       create_labels: bool = True,
                       create_visuals: bool = True) -> Dict:
        """
        Process entire dataset
        
        Args:
            image_dir: Directory with images
            output_base_dir: Base directory for outputs
            conf_threshold: Confidence threshold
            create_labels: Whether to create label files
            create_visuals: Whether to create visual outputs
        
        Returns:
            Processing statistics
        """
        image_dir = Path(image_dir)
        output_base_dir = Path(output_base_dir)
        
        # Create output directories
        labels_dir = output_base_dir / 'labels'
        visuals_dir = output_base_dir / 'visuals'
        
        if create_labels:
            labels_dir.mkdir(parents=True, exist_ok=True)
        if create_visuals:
            visuals_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        print(f"\nProcessing {len(image_files)} images with {self.model_name}...")
        
        stats = {
            'total_images': len(image_files),
            'total_detections': 0,
            'class_counts': {name: 0 for name in self.CLASS_NAMES.values()},
            'processing_time': 0
        }
        
        start_time = time.time()
        
        for img_path in tqdm(image_files, desc="Processing"):
            # Create labels
            if create_labels:
                result = self.create_labels_only(str(img_path), str(labels_dir), conf_threshold)
                stats['total_detections'] += result['num_detections']
                for det in result['detections']:
                    stats['class_counts'][det['class_name']] += 1
            
            # Create visuals
            if create_visuals:
                self.create_labeled_image(str(img_path), str(visuals_dir), conf_threshold)
        
        stats['processing_time'] = time.time() - start_time
        
        # Save statistics
        with open(output_base_dir / f'{self.model_name}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ Processing complete!")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Time: {stats['processing_time']:.2f}s")
        print(f"  Class distribution:")
        for cls_name, count in stats['class_counts'].items():
            print(f"    {cls_name}: {count}")
        
        return stats


def label_dataset_with_top_models():
    """Label entire dataset with top 5 models"""
    
    # Top 5 models
    top_models = [
        ('runs/detect/extreme_stable_v1/weights/best.pt', 'extreme_stable_v1'),
        ('runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt', 'cosine_finetune'),
        ('runs/detect/large_precision_v1_scratch/weights/best.pt', 'large_precision'),
        ('runs/detect/extreme_stable_v2_native/weights/best.pt', 'stable_v2_native'),
        ('runs/detect/moderate_balanced_v1/weights/best.pt', 'moderate_balanced'),
    ]
    
    # Dataset path (update as needed)
    image_dir = 'path/to/dataset/images'
    output_base = 'labeled_outputs'
    
    all_stats = {}
    
    for model_path, model_name in top_models:
        print(f"\n{'='*80}")
        print(f"Processing with: {model_name}")
        print(f"{'='*80}")
        
        labeler = FishLabeler(model_path, model_name)
        
        output_dir = Path(output_base) / model_name
        stats = labeler.process_dataset(
            image_dir=image_dir,
            output_base_dir=str(output_dir),
            conf_threshold=0.25,
            create_labels=True,
            create_visuals=True
        )
        
        all_stats[model_name] = stats
    
    # Save combined statistics
    with open(Path(output_base) / 'all_models_stats.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🏆 ALL MODELS PROCESSED")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {output_base}/")
    print(f"  - labels/      : YOLO format label files")
    print(f"  - visuals/     : Labeled images")
    print(f"  - *_stats.json : Per-model statistics")


def create_live_demo():
    """Create live inference demo with webcam or video"""
    
    model_path = 'runs/detect/extreme_stable_v1/weights/best.pt'
    labeler = FishLabeler(model_path, 'Live Demo')
    
    # For webcam
    cap = cv2.VideoCapture(0)
    
    # Or for video file:
    # cap = cv2.VideoCapture('path/to/video.mp4')
    
    print("Press 'q' to quit, 's' to save screenshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = labeler.model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )[0]
        
        # Draw detections
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            color = labeler.CLASS_COLORS.get(cls, (255, 255, 255))
            
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), 
                        color, 2)
            
            label = f"{labeler.CLASS_NAMES[cls]} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (xyxy[0], xyxy[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Show FPS
        cv2.putText(
            frame,
            f"Model: {labeler.model_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )
        
        cv2.imshow('Fish Detection - Live', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f'screenshot_{timestamp}.jpg', frame)
            print(f"Screenshot saved: screenshot_{timestamp}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    
    # Option 1: Label entire dataset with all top models
    label_dataset_with_top_models()
    
    # Option 2: Live demo
    # create_live_demo()
    
    # Option 3: Create comparison grid for specific image
    # labeler = FishLabeler('path/to/model.pt')
    # top_models = [...]  # List of model paths
    # labeler.create_comparison_grid(
    #     'path/to/image.jpg',
    #     top_models,
    #     'comparison.jpg'
    # )
