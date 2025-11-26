# 🎯 Experiment Plan & Training Guidance

## Recommended Experiment Sequence

### Phase 1: Baseline Establishment (Week 1)

**Run 1: YOLOv11n Baseline**
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 100 \
    --batch 16 \
    --name baseline
```

**Objectives**:
- Establish performance baseline
- Validate training pipeline
- Get initial metrics to beat

**Success Criteria**:
- Training completes without errors
- mAP@50 > 55%
- Model converges (loss decreases steadily)

---

### Phase 2: Optimization (Week 2)

**Run 2: Enhanced Augmentation**
```bash
# Modify hyp_fish.yaml first (increase hsv_s to 0.6, hsv_v to 0.3)
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 120 \
    --batch 16 \
    --name enhanced_aug
```

**Objectives**:
- Improve recall through better generalization
- Test augmentation sensitivity

**Compare with Run 1**:
- Recall improvement: Target +5-10%
- Precision: Allow 2-5% decrease
- Generalization: Better performance on test set

---

**Run 3: Original vs Preprocessed Dataset**
```bash
python train_yolo11_fish.py \
    --data dataset_root \
    --model n \
    --epochs 100 \
    --batch 16 \
    --name no_preprocess
```

**Objectives**:
- Quantify preprocessing benefit
- Identify if dehazing/CLAHE helps

**Decision Point**:
- If Run 3 ≈ Run 1: Preprocessing doesn't help much
- If Run 1 >> Run 3: Keep using preprocessed data

---

### Phase 3: Advanced Experiments (Week 3, Optional)

**Run 4: Transfer from Roboflow (if available)**
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 50 \
    --batch 16 \
    --weights roboflow_checkpoint.pt \
    --name transfer
```

**Run 5: YOLOv11s Comparison**
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model s \
    --epochs 100 \
    --batch 8 \
    --name yolo11s_comparison
```

**Objectives**:
- See upper bound of performance
- Compare nano vs small model
- **Note**: YOLOv11s will likely exceed 70 MB, so this is research only

---

## Metrics Interpretation Guide

### What Each Metric Means

**Precision**: Of all detections the model made, how many were correct?
- High precision (80%+): Few false positives
- Low precision (<70%): Many false alarms

**Recall**: Of all actual fish in images, how many did the model find?
- High recall (70%+): Finds most fish
- Low recall (<50%): Misses many fish

**mAP@50**: Mean Average Precision at IoU threshold 0.5
- Industry standard metric
- Balances precision and recall
- Target: > 60% for production

**mAP@50-95**: Average of mAP at IoU 0.5 to 0.95
- Stricter metric (requires more precise boxes)
- Usually 20-30% lower than mAP@50
- Target: > 40% for production

### Reading Loss Curves

**Box Loss**: How accurate are the bounding box predictions?
- Should decrease steadily
- If it plateaus early, model may be too small

**Class Loss**: How accurate are the class predictions?
- Lower is better
- If high, model struggles to differentiate fish from background

**DFL Loss**: Distribution Focal Loss (YOLO-specific)
- Helps with precise box localization
- Typically smallest of the three losses

### Early Stopping Signals

**Stop training when**:
1. Val loss starts increasing (overfitting)
2. mAP@50 plateaus for 20+ epochs
3. Train loss << Val loss (large gap)

**Continue training when**:
1. Both losses still decreasing
2. mAP still improving
3. No signs of overfitting

---

## Troubleshooting Decision Tree

### Problem: Recall too low (<50%)

**Diagnosis**:
- Model is "too conservative"
- Misses many fish

**Solutions** (try in order):
1. **Inference fix**: Lower `--conf` threshold to 0.15-0.20
2. **Training fix**: Increase `cls` loss weight in hyp_fish.yaml to 0.7
3. **Data fix**: Add more training examples of hard-to-detect fish
4. **Augmentation fix**: Use stronger augmentation (Run 2)

---

### Problem: Precision too low (<70%)

**Diagnosis**:
- Too many false positives
- Model detects non-fish objects

**Solutions** (try in order):
1. **Inference fix**: Increase `--conf` threshold to 0.35-0.40
2. **Data fix**: Review and clean mislabeled training data
3. **Training fix**: Add hard negative examples (images with no fish)
4. **NMS fix**: Use stricter NMS with `--iou 0.35`

---

### Problem: Model size > 70 MB

**Diagnosis**:
- Using wrong model size or export format

**Solutions**:
1. Ensure using YOLOv11n (not s or m)
2. Export with FP16: `--half`
3. Use INT8 quantization: `--int8`
4. Export to TFLite instead of PyTorch

**Expected sizes**:
- YOLOv11n .pt: ~5-8 MB
- YOLOv11n ONNX FP16: ~3-5 MB
- YOLOv11n TFLite INT8: ~2-3 MB
- YOLOv11s .pt: ~20-25 MB (may exceed limit)

---

### Problem: Slow training

**Solutions**:
- Reduce `--batch` size
- Reduce `--workers` to 4
- Use smaller `--imgsz` (640 instead of 768)
- Ensure GPU is being used (`--device 0`)
- Disable caching (already default in script)

---

## Hyperparameter Tuning Guide

### If you want higher recall:

Modify `hyp_fish.yaml`:
```yaml
cls: 0.7          # Increase from 0.5
box: 7.0          # Slightly decrease
mosaic: 1.0       # Keep enabled
mixup: 0.1        # Try adding mixup
```

### If you want higher precision:

Modify `hyp_fish.yaml`:
```yaml
cls: 0.4          # Decrease from 0.5
box: 8.0          # Increase
close_mosaic: 20  # Disable mosaic earlier
```

### If training is unstable:

Modify `hyp_fish.yaml`:
```yaml
lr0: 0.005        # Reduce learning rate
warmup_epochs: 5  # Longer warmup
momentum: 0.9     # Slightly lower momentum
```

---

## Final Recommendations

### For Best Production Model:

1. **Run baseline** (Run 1) to establish metrics
2. **Compare preprocessing** (Run 3) to validate pipeline
3. **Optimize for recall** (Run 2) since recall was weak (43.4%)
4. **Export best model** to ONNX FP16 or TFLite INT8
5. **Validate on test set** before deployment

### Expected Final Metrics:

With good training on this dataset, you should achieve:
- **mAP@50**: 65-75% (improvement over 60.8%)
- **Precision**: 75-85% (slight decrease from 78.4% acceptable)
- **Recall**: 55-65% (major improvement over 43.4%)
- **Model size**: 3-5 MB (ONNX FP16)
- **Inference speed**: 20-30 FPS on Jetson Nano

### Key Success Factors:

1. ✅ Use YOLOv11n for edge deployment
2. ✅ Conservative augmentation (no vertical flips!)
3. ✅ Focus on recall (missing fish is worse than false positives)
4. ✅ Validate on actual edge hardware before deployment
5. ✅ Monitor training curves closely for overfitting
