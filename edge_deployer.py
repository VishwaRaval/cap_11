#!/usr/bin/env python3
"""
Edge Device Deployment Script
Export and optimize YOLOv11 models for edge devices
"""

import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
import time
import os

class EdgeDeployer:
    """Deploy YOLO models to edge devices"""
    
    def __init__(self, model_path: str, model_name: str = None):
        """
        Initialize deployer
        
        Args:
            model_path: Path to .pt model file
            model_name: Optional name for the model
        """
        self.model_path = Path(model_path)
        self.model = YOLO(str(model_path))
        self.model_name = model_name or self.model_path.parent.parent.name
        
        print(f"✓ Loaded model: {self.model_name}")
        print(f"  Model path: {model_path}")
        
        # Get model size
        self.model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model size: {self.model_size_mb:.2f} MB")
    
    def export_onnx(self, output_dir: str, opset: int = 12, 
                    simplify: bool = True, dynamic: bool = False) -> Dict:
        """
        Export to ONNX format
        
        Args:
            output_dir: Directory to save exported model
            opset: ONNX opset version
            simplify: Whether to simplify the model
            dynamic: Whether to use dynamic input shapes
        
        Returns:
            Export statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting to ONNX (opset={opset})...")
        
        start_time = time.time()
        
        # Export
        export_path = self.model.export(
            format='onnx',
            opset=opset,
            simplify=simplify,
            dynamic=dynamic
        )
        
        export_time = time.time() - start_time
        
        # Get exported model size
        export_size = os.path.getsize(export_path) / (1024 * 1024)
        
        # Move to output directory
        final_path = output_dir / f"{self.model_name}.onnx"
        os.rename(export_path, final_path)
        
        stats = {
            'format': 'ONNX',
            'path': str(final_path),
            'original_size_mb': self.model_size_mb,
            'exported_size_mb': export_size,
            'compression_ratio': self.model_size_mb / export_size if export_size > 0 else 0,
            'export_time_s': export_time,
            'opset': opset,
            'simplified': simplify,
            'dynamic': dynamic
        }
        
        print(f"  ✓ Exported: {final_path}")
        print(f"  Size: {export_size:.2f} MB (ratio: {stats['compression_ratio']:.2f}x)")
        print(f"  Time: {export_time:.2f}s")
        
        return stats
    
    def export_tflite(self, output_dir: str, int8: bool = False, 
                      fp16: bool = False) -> Dict:
        """
        Export to TensorFlow Lite format
        
        Args:
            output_dir: Directory to save exported model
            int8: Whether to use INT8 quantization
            fp16: Whether to use FP16 quantization
        
        Returns:
            Export statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        quant_type = 'int8' if int8 else ('fp16' if fp16 else 'fp32')
        print(f"\nExporting to TFLite ({quant_type})...")
        
        start_time = time.time()
        
        # Export
        export_path = self.model.export(
            format='tflite',
            int8=int8,
            half=fp16
        )
        
        export_time = time.time() - start_time
        
        # Get exported model size
        export_size = os.path.getsize(export_path) / (1024 * 1024)
        
        # Move to output directory
        final_path = output_dir / f"{self.model_name}_{quant_type}.tflite"
        os.rename(export_path, final_path)
        
        stats = {
            'format': 'TFLite',
            'path': str(final_path),
            'original_size_mb': self.model_size_mb,
            'exported_size_mb': export_size,
            'compression_ratio': self.model_size_mb / export_size if export_size > 0 else 0,
            'export_time_s': export_time,
            'quantization': quant_type
        }
        
        print(f"  ✓ Exported: {final_path}")
        print(f"  Size: {export_size:.2f} MB (ratio: {stats['compression_ratio']:.2f}x)")
        print(f"  Time: {export_time:.2f}s")
        
        return stats
    
    def export_coreml(self, output_dir: str, nms: bool = True) -> Dict:
        """
        Export to CoreML format (for iOS/macOS)
        
        Args:
            output_dir: Directory to save exported model
            nms: Whether to include NMS in the model
        
        Returns:
            Export statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting to CoreML...")
        
        start_time = time.time()
        
        # Export
        export_path = self.model.export(
            format='coreml',
            nms=nms
        )
        
        export_time = time.time() - start_time
        
        # CoreML exports as a package/directory
        if os.path.isdir(export_path):
            # Get total size of directory
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(export_path)
                for filename in filenames
            ) / (1024 * 1024)
        else:
            total_size = os.path.getsize(export_path) / (1024 * 1024)
        
        # Move to output directory
        final_path = output_dir / f"{self.model_name}.mlpackage"
        if os.path.exists(final_path):
            import shutil
            shutil.rmtree(final_path)
        os.rename(export_path, final_path)
        
        stats = {
            'format': 'CoreML',
            'path': str(final_path),
            'original_size_mb': self.model_size_mb,
            'exported_size_mb': total_size,
            'compression_ratio': self.model_size_mb / total_size if total_size > 0 else 0,
            'export_time_s': export_time,
            'nms_included': nms
        }
        
        print(f"  ✓ Exported: {final_path}")
        print(f"  Size: {total_size:.2f} MB (ratio: {stats['compression_ratio']:.2f}x)")
        print(f"  Time: {export_time:.2f}s")
        
        return stats
    
    def export_torchscript(self, output_dir: str) -> Dict:
        """
        Export to TorchScript format
        
        Args:
            output_dir: Directory to save exported model
        
        Returns:
            Export statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting to TorchScript...")
        
        start_time = time.time()
        
        # Export
        export_path = self.model.export(format='torchscript')
        
        export_time = time.time() - start_time
        
        # Get exported model size
        export_size = os.path.getsize(export_path) / (1024 * 1024)
        
        # Move to output directory
        final_path = output_dir / f"{self.model_name}.torchscript"
        os.rename(export_path, final_path)
        
        stats = {
            'format': 'TorchScript',
            'path': str(final_path),
            'original_size_mb': self.model_size_mb,
            'exported_size_mb': export_size,
            'compression_ratio': self.model_size_mb / export_size if export_size > 0 else 0,
            'export_time_s': export_time
        }
        
        print(f"  ✓ Exported: {final_path}")
        print(f"  Size: {export_size:.2f} MB (ratio: {stats['compression_ratio']:.2f}x)")
        print(f"  Time: {export_time:.2f}s")
        
        return stats
    
    def export_all(self, output_dir: str) -> Dict:
        """
        Export to all formats
        
        Args:
            output_dir: Base directory to save exported models
        
        Returns:
            Combined export statistics
        """
        output_dir = Path(output_dir) / self.model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"EXPORTING MODEL: {self.model_name}")
        print(f"Original size: {self.model_size_mb:.2f} MB")
        print(f"{'='*80}")
        
        all_stats = {
            'model_name': self.model_name,
            'original_path': str(self.model_path),
            'original_size_mb': self.model_size_mb,
            'exports': []
        }
        
        # ONNX
        try:
            stats = self.export_onnx(output_dir)
            all_stats['exports'].append(stats)
        except Exception as e:
            print(f"  ✗ ONNX export failed: {e}")
        
        # TFLite variants
        for int8, fp16, name in [(False, False, 'fp32'), (True, False, 'int8'), (False, True, 'fp16')]:
            try:
                stats = self.export_tflite(output_dir, int8=int8, fp16=fp16)
                all_stats['exports'].append(stats)
            except Exception as e:
                print(f"  ✗ TFLite {name} export failed: {e}")
        
        # CoreML
        try:
            stats = self.export_coreml(output_dir)
            all_stats['exports'].append(stats)
        except Exception as e:
            print(f"  ✗ CoreML export failed: {e}")
        
        # TorchScript
        try:
            stats = self.export_torchscript(output_dir)
            all_stats['exports'].append(stats)
        except Exception as e:
            print(f"  ✗ TorchScript export failed: {e}")
        
        # Save statistics
        stats_path = output_dir / 'export_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"EXPORT SUMMARY")
        print(f"{'='*80}")
        print(f"Original size: {self.model_size_mb:.2f} MB")
        print(f"\nExported formats:")
        for export in all_stats['exports']:
            print(f"  {export['format']:12s}: {export['exported_size_mb']:6.2f} MB "
                  f"({export['compression_ratio']:.2f}x compression)")
        
        # Find best compression
        best = min(all_stats['exports'], key=lambda x: x['exported_size_mb'])
        print(f"\n🏆 Best compression: {best['format']} at {best['exported_size_mb']:.2f} MB")
        
        # Check if under 70MB constraint
        under_70mb = [e for e in all_stats['exports'] if e['exported_size_mb'] < 70]
        if under_70mb:
            print(f"\n✓ Formats under 70MB constraint: {len(under_70mb)}/{len(all_stats['exports'])}")
            for export in under_70mb:
                print(f"  - {export['format']}: {export['exported_size_mb']:.2f} MB")
        else:
            print(f"\n⚠ No formats under 70MB constraint!")
        
        return all_stats
    
    def benchmark_inference(self, image_path: str, num_runs: int = 100) -> Dict:
        """
        Benchmark inference speed
        
        Args:
            image_path: Path to test image
            num_runs: Number of inference runs
        
        Returns:
            Benchmark statistics
        """
        print(f"\nBenchmarking inference on {num_runs} runs...")
        
        # Warmup
        for _ in range(10):
            self.model.predict(image_path, verbose=False)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.model.predict(image_path, verbose=False)
            times.append(time.time() - start)
        
        times = np.array(times)
        
        stats = {
            'num_runs': num_runs,
            'mean_ms': float(times.mean() * 1000),
            'std_ms': float(times.std() * 1000),
            'min_ms': float(times.min() * 1000),
            'max_ms': float(times.max() * 1000),
            'fps': float(1.0 / times.mean())
        }
        
        print(f"  Mean: {stats['mean_ms']:.2f} ms ± {stats['std_ms']:.2f} ms")
        print(f"  FPS: {stats['fps']:.2f}")
        
        return stats


def deploy_top_models():
    """Deploy top 5 models to all edge formats"""
    
    # Top 5 models
    top_models = [
        'runs/detect/extreme_stable_v1/weights/best.pt',
        'runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt',
        'runs/detect/large_precision_v1_scratch/weights/best.pt',
        'runs/detect/extreme_stable_v2_native/weights/best.pt',
        'runs/detect/moderate_balanced_v1/weights/best.pt',
    ]
    
    output_base = 'edge_deployments'
    all_results = []
    
    for model_path in top_models:
        if not Path(model_path).exists():
            print(f"⚠ Model not found: {model_path}")
            continue
        
        deployer = EdgeDeployer(model_path)
        results = deployer.export_all(output_base)
        all_results.append(results)
    
    # Save combined results
    with open(Path(output_base) / 'all_exports.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🏆 ALL MODELS DEPLOYED")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {output_base}/")


def create_deployment_guide():
    """Create deployment guide markdown"""
    
    guide = """# Fish Detection Model Deployment Guide

## 📱 Edge Device Deployment

### Available Formats

1. **ONNX** (.onnx)
   - Best for: Cross-platform deployment, server inference
   - Frameworks: ONNX Runtime, OpenCV DNN, TensorRT
   - Typical size: ~40-60 MB

2. **TensorFlow Lite** (.tflite)
   - Best for: Android, embedded Linux, microcontrollers
   - Variants:
     - FP32: Full precision (~50-70 MB)
     - FP16: Half precision (~25-35 MB) - **RECOMMENDED**
     - INT8: 8-bit quantized (~15-20 MB) - smallest, slight accuracy drop
   - Frameworks: TFLite Runtime

3. **CoreML** (.mlpackage)
   - Best for: iOS and macOS devices
   - Size: ~40-60 MB
   - Optimized for Apple Neural Engine

4. **TorchScript** (.torchscript)
   - Best for: PyTorch-based deployments
   - Size: Similar to original model

### Platform-Specific Recommendations

#### Android
```python
# Use TFLite with FP16 quantization
deployer = EdgeDeployer('model.pt')
deployer.export_tflite(output_dir='android/', fp16=True)
```

#### iOS/macOS
```python
# Use CoreML
deployer = EdgeDeployer('model.pt')
deployer.export_coreml(output_dir='ios/')
```

#### Raspberry Pi / Jetson Nano
```python
# Use ONNX or TFLite INT8
deployer = EdgeDeployer('model.pt')
deployer.export_onnx(output_dir='embedded/')
# or
deployer.export_tflite(output_dir='embedded/', int8=True)
```

#### Web (Browser)
```python
# Use ONNX with ONNX.js
deployer = EdgeDeployer('model.pt')
deployer.export_onnx(output_dir='web/', simplify=True)
```

### Size Optimization

To stay under 70MB constraint:
1. **Use FP16 quantization** - typically 50% size reduction
2. **Use INT8 quantization** - 75% size reduction (small accuracy drop)
3. **Model pruning** - remove less important weights
4. **Knowledge distillation** - train smaller model to mimic larger one

### Inference Code Examples

#### Python (ONNX Runtime)
```python
import onnxruntime as ort
import cv2
import numpy as np

session = ort.InferenceSession("model.onnx")
image = cv2.imread("fish.jpg")
image = cv2.resize(image, (640, 640))
image = image.transpose(2, 0, 1) / 255.0
image = np.expand_dims(image, axis=0).astype(np.float32)

outputs = session.run(None, {"images": image})
```

#### Python (TFLite)
```python
import tensorflow as tf
import cv2
import numpy as np

interpreter = tf.lite.Interpreter(model_path="model_fp16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = cv2.imread("fish.jpg")
image = cv2.resize(image, (640, 640))
image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

#### Swift (CoreML - iOS)
```swift
import CoreML
import Vision

let model = try! YourModelName(configuration: MLModelConfiguration())
let vnModel = try! VNCoreMLModel(for: model.model)

let request = VNCoreMLRequest(model: vnModel) { request, error in
    guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
    // Process results
}

let handler = VNImageRequestHandler(cgImage: yourImage)
try! handler.perform([request])
```

### Performance Benchmarks

Expected inference times on various devices:
- iPhone 13: ~15-20ms (CoreML)
- Pixel 6: ~20-30ms (TFLite FP16)
- Raspberry Pi 4: ~80-120ms (TFLite INT8)
- Jetson Nano: ~30-50ms (ONNX/TensorRT)

### Deployment Checklist

- [ ] Export model to target format
- [ ] Test inference on target device
- [ ] Verify accuracy on test set
- [ ] Measure inference latency
- [ ] Check memory usage
- [ ] Optimize batch size
- [ ] Implement preprocessing pipeline
- [ ] Add error handling
- [ ] Test edge cases (low light, occlusion, etc.)

"""
    
    with open('DEPLOYMENT_GUIDE.md', 'w') as f:
        f.write(guide)
    
    print("✓ Deployment guide created: DEPLOYMENT_GUIDE.md")


if __name__ == "__main__":
    # Deploy all top models
    deploy_top_models()
    
    # Create deployment guide
    create_deployment_guide()
