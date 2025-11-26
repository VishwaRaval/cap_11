#!/usr/bin/env python3
"""
Export YOLOv11 model to edge-optimized formats.

Supports:
- ONNX (FP32, FP16)
- TensorRT (FP16, INT8) - if available
- TFLite (FP16, INT8) - post-training quantization

Target: < 70 MB model size for edge deployment

Usage:
    # Export to ONNX FP16
    python export_edge_model.py --weights runs/detect/fish_n/weights/best.pt --format onnx --half

    # Export to TFLite with INT8 quantization
    python export_edge_model.py --weights runs/detect/fish_n/weights/best.pt --format tflite --int8

    # Export to multiple formats
    python export_edge_model.py --weights runs/detect/fish_n/weights/best.pt --format onnx tflite --half
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import os


def get_file_size_mb(file_path):
    """Get file size in MB."""
    if Path(file_path).exists():
        size_bytes = Path(file_path).stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0


def export_model(weights_path, formats, imgsz=768, half=False, int8=False, simplify=True):
    """
    Export YOLOv11 model to specified formats.
    
    Args:
        weights_path: Path to trained .pt weights
        formats: List of export formats ('onnx', 'tflite', 'engine', etc.)
        imgsz: Input image size
        half: Use FP16 precision
        int8: Use INT8 quantization (for TFLite/TensorRT)
        simplify: Simplify ONNX model
    
    Returns:
        Dictionary of exported model paths and sizes
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    print("=" * 70)
    print("YOLOv11 Edge Model Export")
    print("=" * 70)
    print(f"Input weights: {weights_path}")
    print(f"Original size: {get_file_size_mb(weights_path):.2f} MB")
    print(f"Target size: < 70 MB")
    print(f"Formats: {', '.join(formats)}")
    print(f"Image size: {imgsz}")
    print(f"Half precision (FP16): {half}")
    print(f"INT8 quantization: {int8}")
    print("=" * 70 + "\n")
    
    # Load model
    model = YOLO(weights_path)
    
    exported_models = {}
    
    for fmt in formats:
        print(f"\n🔄 Exporting to {fmt.upper()}...")
        
        try:
            if fmt == 'onnx':
                # Export to ONNX
                export_path = model.export(
                    format='onnx',
                    imgsz=imgsz,
                    half=half,
                    simplify=simplify,
                    dynamic=False,  # Static shape for edge deployment
                    opset=12,
                )
                
            elif fmt == 'tflite':
                # Export to TFLite with quantization
                export_path = model.export(
                    format='tflite',
                    imgsz=imgsz,
                    half=half if not int8 else False,
                    int8=int8,
                )
                
            elif fmt == 'engine' or fmt == 'tensorrt':
                # Export to TensorRT (requires TensorRT installation)
                export_path = model.export(
                    format='engine',
                    imgsz=imgsz,
                    half=half,
                    int8=int8,
                    workspace=4,  # GB
                )
                
            elif fmt == 'torchscript':
                # Export to TorchScript
                export_path = model.export(
                    format='torchscript',
                    imgsz=imgsz,
                )
                
            else:
                print(f"⚠ Warning: Unsupported format '{fmt}', skipping...")
                continue
            
            # Check exported file
            if export_path and Path(export_path).exists():
                file_size = get_file_size_mb(export_path)
                exported_models[fmt] = {
                    'path': str(export_path),
                    'size_mb': file_size
                }
                
                # Check size constraint
                status = "✓" if file_size < 70 else "✗"
                print(f"{status} Exported to: {export_path}")
                print(f"  Size: {file_size:.2f} MB {'(PASS)' if file_size < 70 else '(EXCEEDS 70 MB LIMIT!)'}")
            else:
                print(f"✗ Export failed for {fmt}")
                
        except Exception as e:
            print(f"✗ Error exporting to {fmt}: {e}")
            print(f"  This format may require additional dependencies or hardware support")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    
    if exported_models:
        for fmt, info in exported_models.items():
            status = "✓ PASS" if info['size_mb'] < 70 else "✗ FAIL"
            print(f"{fmt.upper():12s}: {info['size_mb']:8.2f} MB  {status}")
            print(f"             {info['path']}")
        
        # Check if any model meets constraint
        passing_models = [fmt for fmt, info in exported_models.items() if info['size_mb'] < 70]
        if passing_models:
            print(f"\n✓ {len(passing_models)} model(s) meet the < 70 MB constraint")
        else:
            print(f"\n⚠ WARNING: No exported models are under 70 MB!")
            print("  Consider using YOLOv11n (nano) or increasing quantization")
    else:
        print("✗ No models were successfully exported")
    
    print("=" * 70)
    
    return exported_models


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv11 model to edge-optimized formats"
    )
    
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained YOLOv11 weights (.pt file)')
    parser.add_argument('--format', type=str, nargs='+',
                       default=['onnx'],
                       choices=['onnx', 'tflite', 'engine', 'tensorrt', 'torchscript'],
                       help='Export format(s). Options: onnx, tflite, engine (TensorRT), torchscript')
    parser.add_argument('--imgsz', type=int, default=768,
                       help='Input image size (default: 768)')
    parser.add_argument('--half', action='store_true',
                       help='Use FP16 half precision (recommended for edge devices)')
    parser.add_argument('--int8', action='store_true',
                       help='Use INT8 quantization (TFLite/TensorRT only)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Do not simplify ONNX model (default: simplify)')
    
    args = parser.parse_args()
    
    # Normalize format names
    formats = []
    for fmt in args.format:
        if fmt == 'tensorrt':
            formats.append('engine')
        else:
            formats.append(fmt)
    
    # Export model
    exported = export_model(
        weights_path=args.weights,
        formats=formats,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        simplify=not args.no_simplify
    )
    
    # Recommendations
    if exported:
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 70)
        print("For edge deployment:")
        print("  • ONNX FP16: Best compatibility, good performance")
        print("  • TFLite INT8: Smallest size, fastest on mobile/edge TPUs")
        print("  • TensorRT FP16/INT8: Best performance on NVIDIA devices (Jetson)")
        print("\nNext steps:")
        print("  1. Test inference with: python infer_edge.py --model <exported_model>")
        print("  2. Validate accuracy on test set")
        print("  3. Benchmark inference speed on target hardware")
        print("-" * 70)


if __name__ == '__main__':
    main()