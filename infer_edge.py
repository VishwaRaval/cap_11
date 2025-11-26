#!/usr/bin/env python3
"""
Inference script for edge-deployed YOLOv11 models.

Supports:
- PyTorch .pt models
- ONNX models
- TFLite models (via Ultralytics)
- Single image or batch inference
- Visualization and JSON output

Usage:
    # Infer single image
    python infer_edge.py --model runs/detect/fish_n/weights/best.pt --source test_image.jpg

    # Infer on folder with ONNX model
    python infer_edge.py --model best.onnx --source test_images/ --save-json

    # Infer with confidence threshold
    python infer_edge.py --model best.pt --source test.jpg --conf 0.25 --iou 0.45
"""

import argparse
from pathlib import Path
import json
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


def draw_detections(image, boxes, class_names, confidences, thickness=2):
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (BGR)
        boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
        class_names: List of class names
        confidences: List of confidence scores
        thickness: Box line thickness
    
    Returns:
        Image with drawn boxes
    """
    img_draw = image.copy()
    
    for box, cls_name, conf in zip(boxes, class_names, confidences):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw box
        color = (0, 255, 0)  # Green for fish
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{cls_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img_draw


def infer_image(model, image_path, conf_threshold=0.25, iou_threshold=0.45, imgsz=768):
    """
    Run inference on a single image.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to image
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        imgsz: Input image size
    
    Returns:
        Dictionary with detection results
    """
    # Run inference
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False
    )
    
    if len(results) == 0:
        return None
    
    result = results[0]
    
    # Extract detections
    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) > 0 else []
    confidences = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else []
    class_ids = result.boxes.cls.cpu().numpy().astype(int) if len(result.boxes) > 0 else []
    
    # Get class names
    class_names = [result.names[int(cls_id)] for cls_id in class_ids]
    
    # Build detection dict
    detections = {
        'image': str(image_path),
        'num_detections': len(boxes),
        'detections': []
    }
    
    for box, conf, cls_id, cls_name in zip(boxes, confidences, class_ids, class_names):
        detections['detections'].append({
            'class_id': int(cls_id),
            'class_name': cls_name,
            'confidence': float(conf),
            'bbox': [float(x) for x in box]  # [x1, y1, x2, y2]
        })
    
    return detections, boxes, class_names, confidences


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with YOLOv11 edge model"
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model (.pt, .onnx, .tflite, etc.)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or directory')
    parser.add_argument('--output', type=str, default='inference_output',
                       help='Output directory for results (default: inference_output)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--imgsz', type=int, default=768,
                       help='Input image size (default: 768)')
    parser.add_argument('--save-img', action='store_true', default=True,
                       help='Save images with detections (default: True)')
    parser.add_argument('--save-json', action='store_true',
                       help='Save detections as JSON')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display images (useful for batch processing)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    model_path = Path(args.model)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Check model size
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Load with Ultralytics
    try:
        model = YOLO(str(model_path))
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get source files
    source_path = Path(args.source)
    
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
    else:
        raise ValueError(f"Invalid source: {source_path}")
    
    print(f"\nProcessing {len(image_files)} image(s)...")
    print("=" * 70)
    
    # Process images
    all_detections = []
    
    for img_path in tqdm(image_files, desc="Inference"):
        # Run inference
        result = infer_image(
            model, img_path,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz
        )
        
        if result is None:
            continue
        
        detections, boxes, class_names, confidences = result
        all_detections.append(detections)
        
        # Print detections
        print(f"\n{img_path.name}: {detections['num_detections']} detection(s)")
        for det in detections['detections']:
            print(f"  {det['class_name']:15s} {det['confidence']:.3f}  {det['bbox']}")
        
        # Save visualization
        if args.save_img and len(boxes) > 0:
            img = cv2.imread(str(img_path))
            img_draw = draw_detections(img, boxes, class_names, confidences)
            
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), img_draw)
            
            # Display if requested
            if not args.no_display and len(image_files) == 1:
                cv2.imshow('Detections', img_draw)
                print(f"\nPress any key to close window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    # Save JSON
    if args.save_json:
        json_path = output_dir / 'detections.json'
        with open(json_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
        print(f"\n✓ Saved detections to: {json_path}")
    
    # Summary
    total_detections = sum(d['num_detections'] for d in all_detections)
    print("\n" + "=" * 70)
    print(f"✓ Inference complete!")
    print(f"  Processed: {len(image_files)} image(s)")
    print(f"  Total detections: {total_detections}")
    if args.save_img:
        print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()