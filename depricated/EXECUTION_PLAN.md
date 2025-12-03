# 🎯 Execution Plan - YOLOv11 Underwater Fish Detection

## Current Dataset Status

Based on your Roboflow project:
- **Total Images**: 3266 images
- **Dataset Split**:
  - Train: 2948 images (90%)
  - Valid: 169 images (5%)
  - Test: 149 images (5%)
- **Current Metrics** (from Roboflow training):
  - mAP@50: 63.6%
  - Precision: 67.4%
  - Recall: 56.8%
- **Preprocessing Applied**:
  - Resize: Stretch to 768×432
  - Auto-Adjust Contrast
  - Brightness: -10% to +10%
  - Bounding Box Noise: Up to 0.1% of pixels
- **Augmentations**:
  - Saturation: -15% to +15%
  - Outputs per training example: 4

## 📦 Prerequisites

### 1. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installations
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import wandb; print(f'Wandb: {wandb.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### 2. Setup Weights & Biases
```bash
# Option A: Login interactively
wandb login

# Option B: Set environment variable (recommended)
export WANDB_API_KEY="0a78f43170a66024d517c69952f9f8671a49b5ad"

# Verify login
wandb verify
```

## 📂 Dataset Preparation

### Step 1: Download Dataset from Roboflow

You should have received your dataset in this structure:
```
dataset_root/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       ├── img_002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Step 2: Verify Dataset Structure
```bash
# Check that data.yaml exists and is properly formatted
cat dataset_root/data.yaml

# Count images in each split
echo "Train images: $(ls dataset_root/train/images/*.jpg 2>/dev/null | wc -l)"
echo "Valid images: $(ls dataset_root/valid/images/*.jpg 2>/dev/null | wc -l)"
echo "Test images: $(ls dataset_root/test/images/*.jpg 2>/dev/null | wc -l)"

# Check a sample label file
head -n 5 dataset_root/train/labels/$(ls dataset_root/train/labels/ | head -1)
```

Expected `data.yaml` format:
```yaml
path: /absolute/path/to/dataset_root
train: train/images
val: valid/images
test: test/images

nc: 1  # number of classes
names: ['fish']  # or your actual class name(s)
```

### Step 3: Verify Weights File

You should have a `weights.pt` file from Roboflow. Check its details:
```bash
# Check file size
ls -lh weights.pt

# Expected: 5-8 MB for YOLOv11n, 20-25 MB for YOLOv11s
```

## 🚀 Training Execution

### Phase 1: Baseline with Roboflow Weights (Transfer Learning)

**Goal**: Establish baseline using your Roboflow-trained weights as starting point

```bash
# Run 1: Baseline with transfer learning
python train_yolo11_fish.py \
    --data dataset_root \
    --model n \
    --epochs 100 \
    --batch 16 \
    --weights weights.pt \
    --name baseline_transfer \
    --wandb-project "underwater-fish-yolo11" \
    --wandb-notes "Baseline: Transfer learning from Roboflow checkpoint, targeting recall improvement from 56.8%"
```

**Expected Outcomes**:
- Faster convergence (since starting from pre-trained weights)
- Target: mAP@50 > 65% (improvement over 63.6%)
- Target: Recall > 60% (improvement over 56.8%)
- Training time: ~1-2 hours on GPU

**What to Monitor**:
- Check W&B dashboard URL printed at start
- Watch validation recall metric closely
- Ensure val_loss doesn't increase (overfitting)

---

### Phase 2: Enhanced Augmentation for Recall Improvement

**Goal**: Improve recall through stronger augmentation

**Step 2.1**: Modify `hyp_fish.yaml`
```bash
# Edit hyp_fish.yaml and update these values:
# hsv_s: 0.6        # Increased from 0.4
# hsv_v: 0.3        # Increased from 0.2
# scale: 0.4        # Increased from 0.3

# Or use this command to create modified version:
cp hyp_fish.yaml hyp_fish_enhanced.yaml
sed -i 's/hsv_s: 0.4/hsv_s: 0.6/' hyp_fish_enhanced.yaml
sed -i 's/hsv_v: 0.2/hsv_v: 0.3/' hyp_fish_enhanced.yaml
sed -i 's/scale: 0.3/scale: 0.4/' hyp_fish_enhanced.yaml

# Copy to main file
cp hyp_fish_enhanced.yaml hyp_fish.yaml
```

**Step 2.2**: Train with enhanced augmentation
```bash
# Run 2: Enhanced augmentation
python train_yolo11_fish.py \
    --data dataset_root \
    --model n \
    --epochs 120 \
    --batch 16 \
    --weights weights.pt \
    --name enhanced_aug_transfer \
    --wandb-project "underwater-fish-yolo11" \
    --wandb-notes "Enhanced augmentation: hsv_s=0.6, hsv_v=0.3, scale=0.4 for better generalization"
```

**Expected Outcomes**:
- Recall: +5-10% improvement (target: 62-66%)
- Precision: May decrease 2-5% (acceptable trade-off)
- Better generalization on test set

---

### Phase 3: Train from Scratch (Comparison)

**Goal**: Compare transfer learning vs. training from COCO weights

```bash
# Run 3: Train from COCO pretrained (no Roboflow weights)
python train_yolo11_fish.py \
    --data dataset_root \
    --model n \
    --epochs 100 \
    --batch 16 \
    --name from_coco \
    --wandb-project "underwater-fish-yolo11" \
    --wandb-notes "Training from COCO pretrained weights for comparison"
```

**Decision Point**:
- If Run 1/2 (with Roboflow weights) ≈ Run 3: Transfer learning didn't help much
- If Run 1/2 >> Run 3: Keep using Roboflow weights for faster convergence

---

### Phase 4: Additional Preprocessing (Optional)

**Goal**: Test if additional underwater-specific preprocessing helps

**Step 4.1**: Apply preprocessing
```bash
# Create preprocessed dataset with dehazing and CLAHE
python preprocess_images.py \
    --input dataset_root \
    --output dataset_root_preprocessed \
    --dehaze --dehaze-strength 1.0 \
    --clahe --clahe-clip 2.0
```

**Step 4.2**: Train on preprocessed data
```bash
# Run 4: With additional preprocessing
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 100 \
    --batch 16 \
    --weights weights.pt \
    --name with_preprocessing \
    --wandb-project "underwater-fish-yolo11" \
    --wandb-notes "Testing dehazing + CLAHE preprocessing on top of Roboflow preprocessing"
```

**Decision Point**:
- If Run 4 > Run 1: Additional preprocessing helps
- If Run 4 ≈ Run 1: Roboflow preprocessing is sufficient

---

### Phase 5: Model Size Comparison (Optional)

**Goal**: See if a larger model improves performance

```bash
# Run 5: YOLOv11s (small) for comparison
python train_yolo11_fish.py \
    --data dataset_root \
    --model s \
    --epochs 100 \
    --batch 8 \
    --name yolo11s_comparison \
    --wandb-project "underwater-fish-yolo11" \
    --wandb-notes "YOLOv11s for performance upper bound - may exceed 70MB limit"
```

**Note**: YOLOv11s will likely exceed 70 MB, so this is for research only to understand the performance ceiling.

---

## 📊 Monitoring Training

### Real-time Monitoring
While training is running, monitor:

1. **W&B Dashboard**: Click the URL printed at training start
   - Live training curves
   - Real-time metrics updates
   - System resource usage

2. **Local Logs**: Training will print metrics every epoch
   ```
   Epoch   train/box_loss   val/box_loss   mAP50   Precision   Recall
   1/100   0.8234          0.7654         0.523   0.687       0.543
   2/100   0.7123          0.7231         0.551   0.702       0.567
   ...
   ```

3. **Results Directory**: Check `runs/detect/{exp_name}/`
   - `results.csv`: All metrics per epoch
   - `metrics_summary.csv`: Clean summary
   - Training curve plots (*.png)

### Key Metrics to Watch

**Priority: RECALL** (currently 56.8%, target: 60-65%)
- Missing fish is worse than false positives for your use case
- If recall is low after 50 epochs, consider:
  - Lowering confidence threshold during validation
  - Increasing augmentation strength
  - Training for more epochs

**Secondary: mAP@50** (currently 63.6%, target: 65-70%)
- Overall performance indicator
- Should steadily increase

**Monitor: Precision** (currently 67.4%, acceptable: 65-75%)
- Can tolerate 2-5% decrease if recall improves significantly

### Early Stopping Indicators

**Stop training if**:
- Val loss increases for 50+ consecutive epochs (patience=50)
- mAP@50 plateaus for 30+ epochs
- Overfitting: large gap between train and val loss

**Continue training if**:
- Both losses still decreasing
- mAP still improving
- Recall hasn't plateaued

---

## 🎯 Model Export & Validation

### After Best Model is Identified

```bash
# Export best model to edge formats
python export_edge_model.py \
    --weights runs/detect/best_run_name/weights/best.pt \
    --format onnx tflite \
    --half \
    --int8

# Expected output:
# ONNX FP16: ~3-5 MB
# TFLite INT8: ~2-3 MB
```

### Validate on Test Set

```bash
# Run inference on test set
python infer_edge.py \
    --model runs/detect/best_run_name/weights/best.pt \
    --source dataset_root/test/images/ \
    --save-json \
    --conf 0.25

# Check detection statistics
cat inference_output/detections.json
```

### Test Different Confidence Thresholds

```bash
# Lower threshold for higher recall
python infer_edge.py \
    --model best.pt \
    --source dataset_root/test/images/ \
    --conf 0.15 \
    --save-json

# Higher threshold for higher precision
python infer_edge.py \
    --model best.pt \
    --source dataset_root/test/images/ \
    --conf 0.35 \
    --save-json

# Compare results to find optimal threshold
```

---

## 📈 Comparing Experiments in W&B

### After Multiple Runs

1. Go to W&B project: https://wandb.ai/your-username/underwater-fish-yolo11
2. Select multiple runs (checkboxes)
3. Click "Compare" button
4. View parallel coordinates plot
5. Filter by tags: `nano`, `edge-deployment`

### Key Comparisons

Compare the following across runs:
- **Recall progression**: Which run reaches highest recall?
- **mAP@50 final value**: Which has best overall performance?
- **Training efficiency**: Which converges fastest?
- **Generalization**: Val vs train loss gap

### Decision Matrix

| Experiment | Best For | Choose If |
|------------|----------|-----------|
| Baseline Transfer | Quick results | Recall > 60%, mAP > 65% |
| Enhanced Aug | Higher recall | Recall > baseline by 3%+ |
| From COCO | Comparison | Curious about transfer learning value |
| With Preprocessing | Challenging conditions | Significant mAP improvement |
| YOLOv11s | Research only | Want to see upper bound |

---

## 🚀 Production Deployment

### Final Model Selection Criteria

Choose model that meets:
1. ✅ mAP@50 > 65%
2. ✅ Recall > 60% (priority!)
3. ✅ Precision > 65%
4. ✅ Exported model < 70 MB
5. ✅ Inference speed > 15 FPS on target hardware

### Deployment Checklist

- [ ] Model exported to target format (ONNX/TFLite)
- [ ] Size verified < 70 MB
- [ ] Tested on sample test images
- [ ] Optimal confidence threshold determined (likely 0.20-0.30)
- [ ] False positive rate acceptable for use case
- [ ] Inference speed benchmarked on edge device
- [ ] Model artifacts saved in W&B
- [ ] Documentation updated with final metrics

---

## 🐛 Troubleshooting

### Issue: Training crashes with OOM (Out of Memory)
```bash
# Solution: Reduce batch size
--batch 8  # or even 4

# Or reduce image size
--imgsz 640
```

### Issue: Recall not improving
```bash
# Solution 1: Lower confidence threshold during inference
python infer_edge.py --model best.pt --source test.jpg --conf 0.15

# Solution 2: Increase cls loss weight in hyp_fish.yaml
cls: 0.7  # increased from 0.5

# Solution 3: Train for more epochs
--epochs 150
```

### Issue: Too many false positives (low precision)
```bash
# Solution 1: Increase confidence threshold
--conf 0.35

# Solution 2: Review training data for mislabeled examples
# Check images with false positives and verify labels

# Solution 3: Use stricter NMS
python infer_edge.py --model best.pt --source test.jpg --iou 0.35
```

### Issue: Model size > 70 MB after export
```bash
# Solution: Use INT8 quantization
python export_edge_model.py \
    --weights best.pt \
    --format tflite \
    --int8

# Or use ONNX FP16
python export_edge_model.py \
    --weights best.pt \
    --format onnx \
    --half
```

### Issue: W&B not logging
```bash
# Check API key
echo $WANDB_API_KEY

# Or pass directly
python train_yolo11_fish.py ... --wandb-key "your_key"

# Test login
wandb login
wandb verify
```

---

## 📝 Experiment Tracking Template

Keep notes for each run:

```markdown
## Run: baseline_transfer
- **Date**: 2024-XX-XX
- **Dataset**: dataset_root (Roboflow preprocessed)
- **Weights**: weights.pt (Roboflow checkpoint)
- **Hyperparameters**: Default from hyp_fish.yaml
- **Results**:
  - Final mAP@50: X.X%
  - Final Recall: X.X%
  - Final Precision: X.X%
  - Training time: X hours
- **Observations**: [Your notes]
- **W&B Link**: https://wandb.ai/...
- **Decision**: [Keep/Improve/Discard]
```

---

## 🎯 Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup & Verification | 30 min | Install deps, verify dataset |
| Run 1: Baseline | 1-2 hours | Transfer from Roboflow weights |
| Run 2: Enhanced Aug | 1.5-2.5 hours | Longer epochs (120) |
| Run 3: From COCO | 1-2 hours | Comparison run |
| Analysis & Comparison | 30 min | Compare in W&B |
| Optional Runs 4-5 | 2-4 hours | If needed |
| Export & Validation | 30 min | Export and test |
| **Total** | **5-9 hours** | Can run overnight |

---

## 🎓 Success Criteria

Your training is successful when:

1. **Performance Targets Met**:
   - ✅ mAP@50: 65-70% (vs current 63.6%)
   - ✅ Recall: 60-65% (vs current 56.8%) ← **PRIORITY**
   - ✅ Precision: 65-75% (vs current 67.4%)

2. **Edge Constraints Met**:
   - ✅ Exported model < 70 MB
   - ✅ Inference speed > 15 FPS on target device

3. **Deployment Ready**:
   - ✅ Model validated on test set
   - ✅ Confidence threshold optimized
   - ✅ False positive rate acceptable
   - ✅ Model artifacts saved and documented

---

## 📚 Quick Reference Commands

```bash
# Full training workflow
pip install -r requirements.txt
export WANDB_API_KEY="0a78f43170a66024d517c69952f9f8671a49b5ad"

# Best starting point (recommended)
python train_yolo11_fish.py \
    --data dataset_root \
    --model n \
    --epochs 100 \
    --batch 16 \
    --weights weights.pt \
    --name baseline_transfer \
    --wandb-project "underwater-fish-yolo11"

# Monitor training
# Check W&B URL printed at start
# Or: tail -f runs/detect/fish_n_baseline_transfer/results.csv

# After training completes
python export_edge_model.py --weights runs/detect/fish_n_baseline_transfer/weights/best.pt --format onnx tflite --half --int8
python infer_edge.py --model runs/detect/fish_n_baseline_transfer/weights/best.pt --source dataset_root/test/images/
```

---

**Good luck with your training! 🐠📊🚀**

**Remember**: Focus on improving recall (currently 56.8%) as it's your main challenge. Missing fish detections is more critical than occasional false positives in your underwater monitoring use case.
