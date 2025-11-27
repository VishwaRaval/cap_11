# 📊 Detailed W&B Metrics Logging Documentation

## Overview

The enhanced training script logs **comprehensive, real-time metrics** to Weights & Biases. This document details every metric tracked during training.

---

## 🎯 Metrics Logged Per Epoch

### 1. Training Losses

| Metric | Description | Good Value | What It Means |
|--------|-------------|------------|---------------|
| `train/box_loss` | Bounding box regression loss | Decreasing, < 0.05 | How well the model predicts box coordinates |
| `train/cls_loss` | Classification loss | Decreasing, < 0.5 | How well the model classifies objects as fish |
| `train/dfl_loss` | Distribution Focal Loss | Decreasing, < 0.5 | Box localization quality (YOLO-specific) |

**What to watch**: All three should decrease steadily. If they plateau early, model may need more capacity or training time.

---

### 2. Validation Losses

| Metric | Description | Good Value | What It Means |
|--------|-------------|------------|---------------|
| `val/box_loss` | Validation box regression loss | < train/box_loss | Box prediction quality on unseen data |
| `val/cls_loss` | Validation classification loss | < train/cls_loss | Classification quality on unseen data |
| `val/dfl_loss` | Validation distribution loss | < train/dfl_loss | Localization quality on unseen data |

**What to watch**: 
- Should track training losses but remain slightly higher
- If val loss >> train loss: Overfitting
- If val loss increases while train loss decreases: Stop training

---

### 3. Performance Metrics (Core)

| Metric | Description | Current Baseline | Target | What It Means |
|--------|-------------|------------------|--------|---------------|
| `metrics/precision` | Precision at confidence threshold | 67.4% | 65-75% | Of all detections, what % are correct? |
| `metrics/recall` | Recall at confidence threshold | **56.8%** | **60-65%** | **Of all fish, what % are detected?** |
| `metrics/mAP50` | Mean Average Precision @ IoU 0.5 | 63.6% | 65-70% | Overall detection performance |
| `metrics/mAP50-95` | mAP averaged over IoU 0.5 to 0.95 | ~40-45% | 40-50% | Stricter performance metric |

**Priority**: **Recall** is your main optimization target (currently 56.8%, targeting 60-65%)

---

### 4. Derived Metrics (Computed)

| Metric | Description | Formula | What It Means |
|--------|-------------|---------|---------------|
| `metrics/f1_score` | F1 Score | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| `metrics/recall_improvement` | Recall improvement over baseline | Recall - 0.568 | Absolute improvement in recall |
| `metrics/recall_improvement_pct` | Recall improvement percentage | (Recall - 0.568) / 0.568 × 100 | % improvement in recall |
| `metrics/map50_improvement` | mAP@50 improvement over baseline | mAP50 - 0.636 | Absolute improvement in mAP@50 |
| `metrics/map50_improvement_pct` | mAP@50 improvement percentage | (mAP50 - 0.636) / 0.636 × 100 | % improvement in mAP@50 |

**Why this matters**: These metrics let you instantly see if you're improving over your Roboflow baseline.

---

### 5. Learning Rate Schedule

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| `lr/param_group_0` | Learning rate for first parameter group | 0.0001 - 0.01 |
| `lr/param_group_1` | Learning rate for second parameter group | 0.0001 - 0.01 |
| `lr/param_group_2` | Learning rate for third parameter group | 0.0001 - 0.01 |

**What to watch**: Should decrease over time (learning rate scheduling)

---

## 📈 Training Visualizations Logged

### Automatic Plots

1. **Training Curves** (`plots/results.png`)
   - Loss curves (train vs val)
   - mAP progression
   - Precision/Recall curves
   - All in one comprehensive plot

2. **Confusion Matrix** (`confusion_matrix`)
   - Shows true positives, false positives, false negatives
   - Helps identify if model confuses fish with background

3. **PR Curve** (`pr_curve`)
   - Precision-Recall curve
   - Shows trade-off at different confidence thresholds
   - Area under curve indicates model quality

4. **F1 Curve** (`f1_curve`)
   - F1 score at different confidence thresholds
   - Helps find optimal confidence threshold

5. **All Other Ultralytics Plots**
   - Batch predictions
   - Label correlations
   - And more...

---

## 🎯 Configuration Parameters Logged

### Model Configuration
- `model_size`: n / s / m
- `architecture`: YOLOv11n / YOLOv11s / YOLOv11m
- `weights_init`: roboflow_transfer or coco_pretrained

### Training Parameters
- `epochs`: Number of training epochs
- `batch_size`: Batch size
- `image_size`: Input image size (768)
- `optimizer`: Optimizer used (auto)
- `learning_rate`: Initial learning rate (0.01)
- `warmup_epochs`: Warmup epochs (3.0)
- `momentum`: SGD momentum (0.937)
- `weight_decay`: Weight decay (0.0005)

### Loss Weights
- `box_loss_gain`: Box loss weight (7.5)
- `cls_loss_gain`: Classification loss weight (0.5)
- `dfl_loss_gain`: DFL loss weight (1.5)

### Augmentation Parameters
- `mosaic`: Mosaic augmentation probability (1.0)
- `mixup`: Mixup augmentation probability (0.0)
- `degrees`: Rotation degrees (5.0)
- `translate`: Translation fraction (0.1)
- `scale`: Scale range (0.3)
- `shear`: Shear degrees (2.0)
- `flipud`: Vertical flip probability (0.0 - disabled for fish!)
- `fliplr`: Horizontal flip probability (0.5)
- `hsv_h`: Hue augmentation (0.01)
- `hsv_s`: Saturation augmentation (0.4)
- `hsv_v`: Value/brightness augmentation (0.2)

### Project Constraints
- `task`: underwater_fish_detection
- `target_deployment`: edge_device
- `size_constraint_mb`: 70
- `target_recall`: 0.60
- `current_baseline_recall`: 0.568
- `current_baseline_map50`: 0.636
- `current_baseline_precision`: 0.674

---

## 📦 Model Artifacts Logged

### What Gets Uploaded

1. **Best Model Weights** (`best.pt`)
   - Model with best validation mAP@50
   - Automatically saved as versioned artifact
   - Can be downloaded directly from W&B

2. **Last Model Weights** (`last.pt`)
   - Model from final training epoch
   - Useful if training was stopped early

3. **Metadata**
   - Model size (n/s/m)
   - File size in MB
   - Framework (ultralytics)
   - Task (object_detection)

---

## 📊 Final Summary Metrics

At the end of training, these summary metrics are logged:

### Final Performance
- `final/train_box_loss`: Final training box loss
- `final/train_cls_loss`: Final training class loss
- `final/train_dfl_loss`: Final training DFL loss
- `final/val_box_loss`: Final validation box loss
- `final/val_cls_loss`: Final validation class loss
- `final/val_dfl_loss`: Final validation DFL loss
- `final/precision`: Final precision
- `final/recall`: Final recall
- `final/mAP50`: Final mAP@50
- `final/mAP50_95`: Final mAP@50-95
- `final/f1_score`: Final F1 score

### Final Improvements
- `final/recall_improvement`: Absolute recall improvement over baseline
- `final/recall_improvement_pct`: % recall improvement
- `final/recall_target_met`: Boolean - Did we reach 60% recall target?
- `final/map50_improvement`: Absolute mAP@50 improvement
- `final/map50_improvement_pct`: % mAP@50 improvement
- `final/map50_target_met`: Boolean - Did we reach 65% mAP@50 target?

---

## 🔍 How to Use These Metrics

### During Training

**Monitor in Real-Time**:
1. Open W&B dashboard (URL printed at training start)
2. Watch these panels:
   - **Recall trend**: Is it increasing toward 60%?
   - **Loss curves**: Are train/val losses both decreasing?
   - **mAP@50 trend**: Is overall performance improving?

**Red Flags**:
- Val loss increases while train loss decreases → **Overfitting**
- Recall plateaus at < 55% for 30+ epochs → **Need stronger augmentation**
- Precision drops below 65% → **Too many false positives**

### After Training

**Compare Experiments**:
1. Go to W&B project page
2. Select multiple runs
3. Click "Compare"
4. Look at:
   - Which run has highest `final/recall`?
   - Which has best `final/recall_improvement_pct`?
   - Which reached `final/recall_target_met = True`?

**Export Data**:
- Download any metric as CSV for custom analysis
- Download model artifacts (best.pt) for deployment
- Generate reports to share results

---

## 📈 Example: Reading Your Dashboard

### Good Training Run Example

```
Epoch 50:
  train/box_loss: 0.0423 ✓ (decreasing)
  val/box_loss: 0.0518 ✓ (slightly higher than train)
  metrics/recall: 0.612 ✓ (above 60% target!)
  metrics/recall_improvement: +0.044 ✓ (+4.4% points)
  metrics/recall_improvement_pct: +7.75% ✓
  metrics/mAP50: 0.671 ✓ (above 65% target!)
  metrics/precision: 0.724 ✓ (acceptable range)
  metrics/f1_score: 0.665 ✓ (good balance)
```

### Problematic Training Run Example

```
Epoch 50:
  train/box_loss: 0.0235 ⚠ (very low)
  val/box_loss: 0.0887 ⚠ (much higher than train - OVERFITTING!)
  metrics/recall: 0.543 ✗ (below baseline - getting worse!)
  metrics/recall_improvement: -0.025 ✗ (negative improvement)
  metrics/precision: 0.823 ⚠ (too high - model too conservative)
```

**Diagnosis**: Overfitting + Model too conservative. Need to:
- Stop training (overfitting)
- Add more augmentation
- Lower confidence threshold during inference

---

## 🎯 Key Metrics to Track

### Priority 1: Recall (Most Important!)
- **Current**: 56.8%
- **Target**: 60-65%
- **Track**: `metrics/recall` and `metrics/recall_improvement`
- **Why**: Missing fish is worse than false positives in your use case

### Priority 2: mAP@50
- **Current**: 63.6%
- **Target**: 65-70%
- **Track**: `metrics/mAP50` and `metrics/map50_improvement`
- **Why**: Overall performance indicator

### Priority 3: Precision
- **Current**: 67.4%
- **Acceptable**: 65-75%
- **Track**: `metrics/precision`
- **Why**: Can tolerate 2-5% decrease if recall improves

### Priority 4: Loss Divergence
- **Track**: `val/box_loss` vs `train/box_loss`
- **Why**: Early indicator of overfitting

---

## 💡 Pro Tips

1. **Use W&B's Compare Feature**
   - Compare recall improvements across runs
   - Filter by tags (e.g., "nano", "enhanced_aug")
   - Create custom views for specific metrics

2. **Set Up Alerts**
   - Alert when `metrics/recall` > 0.60 (target met!)
   - Alert when val loss increases (overfitting)

3. **Download Model Artifacts**
   - Best model is automatically versioned in W&B
   - Download directly for deployment
   - No need to manage local files

4. **Export Plots**
   - Download any plot as high-res image
   - Use in reports or presentations
   - Share with team

5. **Create Reports**
   - W&B Reports feature lets you document findings
   - Combine plots, metrics, and notes
   - Share with collaborators

---

## 🔗 Quick Links

- **W&B Docs**: https://docs.wandb.ai/
- **Metrics API**: https://docs.wandb.ai/ref/python/run#log
- **Artifacts**: https://docs.wandb.ai/guides/artifacts
- **Reports**: https://docs.wandb.ai/guides/reports

---

**Your training script now logs the most comprehensive metrics possible!** 📊✨

Every metric is designed to help you:
1. Monitor training in real-time
2. Diagnose issues quickly
3. Compare experiments effectively
4. Track improvement over baseline
5. Make data-driven decisions
