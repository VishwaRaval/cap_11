# YOLOv11 Underwater Fish Detection - Edge Deployment Pipeline

Complete training and deployment pipeline for underwater fish object detection using YOLOv11, optimized for edge devices (< 70 MB model size).

## 📁 Project Structure
```
.
├── requirements.txt              # Python dependencies
├── preprocess_images.py          # Offline preprocessing with dehazing/CLAHE
├── hyp_fish.yaml                 # Hyperparameters tailored for underwater fish
├── train_yolo11_fish.py          # Main training script
├── export_edge_model.py          # Export to edge formats (ONNX/TFLite/TensorRT)
├── infer_edge.py                 # Inference script
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Verify Ultralytics installation
yolo checks
```

### 2. Dataset Preparation

Your dataset should be in this structure:
```
dataset_root/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 3. Optional Preprocessing

Apply underwater-specific enhancements:
```bash
# With dehazing and CLAHE
python preprocess_images.py \
    --input dataset_root \
    --output dataset_root_preprocessed \
    --dehaze --clahe

# Adjust dehazing strength
python preprocess_images.py \
    --input dataset_root \
    --output dataset_root_preprocessed \
    --dehaze --dehaze-strength 1.5 \
    --clahe --clahe-clip 2.5
```

### 4. Training
```bash
# Train YOLOv11n (nano) - recommended for edge deployment
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 100 \
    --batch 16

# Train YOLOv11s (small) for comparison
python train_yolo11_fish.py \
    --data dataset_root \
    --model s \
    --epochs 100 \
    --batch 8
```

### 5. Export for Edge Deployment
```bash
# Export to ONNX FP16 (recommended)
python export_edge_model.py \
    --weights runs/detect/fish_n/weights/best.pt \
    --format onnx \
    --half

# Export to TFLite INT8 (smallest size)
python export_edge_model.py \
    --weights runs/detect/fish_n/weights/best.pt \
    --format tflite \
    --int8

# Export to multiple formats
python export_edge_model.py \
    --weights runs/detect/fish_n/weights/best.pt \
    --format onnx tflite \
    --half
```

### 6. Inference
```bash
# Single image
python infer_edge.py \
    --model runs/detect/fish_n/weights/best.pt \
    --source test_image.jpg

# Batch inference on folder
python infer_edge.py \
    --model best.onnx \
    --source test_images/ \
    --save-json

# With custom thresholds
python infer_edge.py \
    --model best.pt \
    --source test.jpg \
    --conf 0.3 \
    --iou 0.5
```

## 📊 Experiment Plan

### Experiment A: Baseline (YOLOv11n + Preprocessed Dataset)

**Goal**: Establish baseline with underwater preprocessing
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 100 \
    --batch 16 \
    --name baseline
```

**Expected Outcomes**:
- mAP@50: 60-70% (targeting improvement over 60.8%)
- Precision: 75-85%
- Recall: 50-60% (targeting improvement over 43.4%)
- Training time: ~2-3 hours on single GPU

**What to Watch**:
- Validation mAP@50 should steadily increase
- If val loss plateaus early, consider increasing epochs
- Check for overfitting after epoch 70-80

---

### Experiment B: Enhanced Augmentation

**Goal**: Improve recall through stronger augmentation

Modify `hyp_fish.yaml`:
```yaml
hsv_s: 0.6        # Increased from 0.4
hsv_v: 0.3        # Increased from 0.2
scale: 0.4        # Increased from 0.3
mosaic: 1.0       # Keep enabled
```
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 120 \
    --batch 16 \
    --name enhanced_aug
```

**Expected Outcomes**:
- Recall: +5-10% improvement
- Precision: May decrease slightly (2-5%)
- Better generalization on test set

**Trade-offs**:
- Longer training needed (120 epochs)
- Slightly lower precision acceptable for higher recall

---

### Experiment C: Transfer from Roboflow Checkpoint (if available)

**Goal**: Leverage domain-specific pretrained weights
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 50 \
    --batch 16 \
    --weights path/to/roboflow_checkpoint.pt \
    --name roboflow_transfer
```

**Expected Outcomes**:
- Faster convergence (needs fewer epochs)
- Potentially higher mAP if Roboflow model was well-trained
- Good starting point for fine-tuning

**When to Use**:
- If you have access to Roboflow-trained weights
- When dataset distribution hasn't changed much
- As a warm start for hyperparameter search

---

### Experiment D: Original Dataset (No Preprocessing)

**Goal**: Compare preprocessing effectiveness
```bash
python train_yolo11_fish.py \
    --data dataset_root \
    --model n \
    --epochs 100 \
    --batch 16 \
    --name no_preprocess
```

**Compare with Experiment A**:
- Does dehazing/CLAHE improve metrics?
- Visual quality of detections
- Inference robustness on challenging underwater conditions

---

## 📈 Metrics Interpretation

### Reading Training Metrics

After training, check `runs/detect/{exp_name}/metrics_summary.csv`:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('runs/detect/fish_n_baseline/metrics_summary.csv')

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss curves
axes[0,0].plot(df['epoch'], df['train_box_loss'], label='Train Box Loss')
axes[0,0].plot(df['epoch'], df['val_box_loss'], label='Val Box Loss')
axes[0,0].set_title('Box Loss')
axes[0,0].legend()

# mAP
axes[0,1].plot(df['epoch'], df['mAP50'], label='mAP@50')
axes[0,1].plot(df['epoch'], df['mAP50_95'], label='mAP@50-95')
axes[0,1].set_title('mAP Metrics')
axes[0,1].legend()

# Precision/Recall
axes[1,0].plot(df['epoch'], df['precision'], label='Precision')
axes[1,0].plot(df['epoch'], df['recall'], label='Recall')
axes[1,0].set_title('Precision vs Recall')
axes[1,0].legend()

plt.tight_layout()
plt.savefig('training_curves.png')
```

### Detecting Issues

#### 1. Overfitting
**Symptoms**:
- Val loss increases while train loss decreases
- mAP@50 decreases after initial improvement
- Large gap between train and val metrics

**Solutions**:
- Reduce epochs (use early stopping)
- Increase augmentation strength
- Add dropout (modify model architecture)
- Use more training data

#### 2. Underfitting
**Symptoms**:
- Both train and val loss remain high
- mAP plateaus at low value (< 40%)
- Metrics don't improve after 50+ epochs

**Solutions**:
- Increase model capacity (use YOLOv11s instead of n)
- Reduce augmentation strength
- Increase learning rate
- Train for more epochs

#### 3. Low Recall / High Precision
**Symptoms**:
- Precision: 80-90%
- Recall: < 50%
- Model is "too conservative" (misses many fish)

**Solutions**:
- Lower confidence threshold during inference (--conf 0.15)
- Increase `cls` loss weight in `hyp_fish.yaml`
- Add more positive examples to training data
- Stronger augmentation to see fish in more contexts

#### 4. Low Precision / High Recall
**Symptoms**:
- Precision: < 70%
- Recall: 70-80%
- Many false positives (detecting non-fish objects)

**Solutions**:
- Increase confidence threshold (--conf 0.35)
- Clean training data (remove mislabeled examples)
- Reduce augmentation that creates unrealistic images
- Increase `box` loss weight

### Target Metrics for Production

For underwater fish detection edge deployment:

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| mAP@50 | > 70% | 60-70% | < 60% |
| Precision | > 80% | 70-80% | < 70% |
| Recall | > 60% | 50-60% | < 50% |
| Inference Speed | > 20 FPS | 10-20 FPS | < 10 FPS |
| Model Size | < 20 MB | 20-50 MB | > 50 MB |

## 🎯 Optimization Tips

### For Better Recall:
1. Lower confidence threshold in inference
2. Increase training epochs
3. Use stronger augmentation
4. Add hard negative mining
5. Adjust class weights if imbalanced

### For Better Precision:
1. Clean training labels carefully
2. Increase confidence threshold
3. Use harder NMS (lower IoU threshold)
4. Reduce false positive triggers
5. Add negative samples (backgrounds)

### For Faster Inference:
1. Use YOLOv11n (not s or m)
2. Export to TFLite INT8
3. Reduce input size (640 instead of 768)
4. Use TensorRT on NVIDIA devices
5. Enable GPU/NPU acceleration on edge device

### For Smaller Model:
1. Always use YOLOv11n
2. Export with INT8 quantization
3. Prune unnecessary layers (advanced)
4. Use knowledge distillation (advanced)

## 🐛 Troubleshooting

### Issue: Model size > 70 MB after export

**Solution**:
```bash
# Use INT8 quantization
python export_edge_model.py \
    --weights best.pt \
    --format tflite \
    --int8

# Or use FP16 ONNX
python export_edge_model.py \
    --weights best.pt \
    --format onnx \
    --half
```

### Issue: Low mAP on validation set

**Possible causes**:
1. Dataset too small → Use augmentation
2. Poor quality labels → Clean dataset
3. Model too small → Try YOLOv11s
4. Insufficient training → Increase epochs

### Issue: Slow inference on edge device

**Solutions**:
1. Reduce input size (640×384 instead of 768×432)
2. Use INT8 quantized TFLite
3. Enable hardware acceleration (GPU/NPU)
4. Batch multiple images if possible

### Issue: Training crashes (OOM)

**Solutions**:
```bash
# Reduce batch size
--batch 8  # or even 4

# Reduce image size
--imgsz 640

# Disable caching
# (automatically done in train script)

# Use gradient accumulation (Ultralytics auto-handles this)
```

## 📚 Additional Resources

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com)
- [YOLO Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)
- [Edge Deployment Guide](https://docs.ultralytics.com/guides/edge-deployment/)

## 🔍 Model Performance Checklist

Before deploying to edge device:

- [ ] mAP@50 > 60% on test set
- [ ] Model size < 70 MB
- [ ] Inference tested on sample images
- [ ] Confidence threshold tuned for use case
- [ ] False positive rate acceptable
- [ ] Inference speed measured on target hardware
- [ ] Model exported to appropriate format (ONNX/TFLite)
- [ ] Quantization applied if needed
- [ ] Visual inspection of predictions looks good

## 📝 Notes

- **No vertical flips**: Critical for underwater scenes (fish don't swim upside down)
- **Conservative augmentation**: Dataset already preprocessed by Roboflow
- **Focus on recall**: Better to detect more fish (even with some FP) than miss fish
- **Edge constraints**: 70 MB limit requires nano model + quantization
- **Preprocessing**: Dehazing/CLAHE optional but recommended for murky water

---

**Good luck with your underwater fish detection project! 🐠**