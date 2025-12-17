# 🐠 Underwater Fish Detection with YOLOv11

**A comprehensive computer vision system for real-time underwater fish species classification, achieving 70.11% accuracy through ensemble methods while maintaining edge deployment constraints.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Project Overview

This capstone project developed a production-ready underwater fish detection system using YOLOv11, optimized for edge device deployment. Through 60+ systematic experiments over 6 weeks, we achieved **70.11% accuracy** (exceeding the 70% target) using a 5-model ensemble while maintaining a <70MB per-model size constraint.

### 🎯 Key Achievements

- ✅ **70.11% Accuracy** (66.27% precision, 73.94% recall) - Exceeds 70% target
- ✅ **5-Model Ensemble** with optimized confidence threshold (0.45)
- ✅ **Edge-Deployable** - 18.3 MB per model (<70MB constraint)
- ✅ **60+ Experiments** - Systematic hyperparameter optimization
- ✅ **Comprehensive Documentation** - All successes and failures documented
- ✅ **Production Tools** - Complete labeling and deployment pipeline

### 🐟 Fish Species Detected

| Species | Dataset % | Instances | Challenge |
|---------|-----------|-----------|-----------|
| Surgeon Fish | 62.8% | 4,924 | Majority class |
| Grunt Fish | 30.0% | 2,348 | Balanced |
| Parrot Fish | 7.2% | 564 | **Minority class** (8.73:1 imbalance) |

---

## 📊 Performance Summary

### Final Ensemble Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | **70.11%** | (Precision + Recall) / 2 |
| Precision | 66.27% | Optimized for recall |
| Recall | 73.94% | Prioritizes catching fish |
| F1 Score | 69.89% | Balanced performance |
| Confidence Threshold | 0.45 | Optimized through sweep |

### Single Model Performance

| Model | Precision | Recall | Avg Acc | Notes |
|-------|-----------|--------|---------|-------|
| exp3_lab_aggressive_conservative | 68.87% | 62.58% | 65.73% | Best single ✓ |
| ultra_stable_v1 | 62.60% | 68.60% | 65.60% | High recall |
| extreme_stable_v1 | 63.73% | 66.15% | 64.94% | Balanced |

**Single-Model Ceiling**: 65-67% (data quality limitation)  
**Ensemble Improvement**: +4.38% over best single model

---

## 🗂️ Project Structure

```
.
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── FISH_DETECTION_PROJECT_REPORT.md  # Complete technical report
│
├── 📂 Training Scripts (Active)
│   ├── train_ultra_stable.py                   # Ultra-conservative training (best approach)
│   ├── train_ultra_stable_v2.py                # Multi-checkpoint saving version
│   └── train_native_resolution.py              # Aspect-ratio preserving training
│
├── 📂 Hyperparameter Configurations
│   ├── hyp_fish_extreme_stable.yaml            # Best single model config (lr=0.00005)
│   ├── hyp_fish_ultra_stable.yaml              # Workhorse config (lr=0.0001)
│   └── hyp_fish_moderate.yaml                  # Balanced approach
│
├── 📂 Ensemble & Inference
│   ├── ensemble_with_tta.py                    # Ensemble with test-time augmentation
│   ├── true_ensemble_inference.py              # Production ensemble inference
│   └── label_dataset_with_ensemble.py          # 3-mode dataset labeling system
│
├── 📂 Deployment
│   ├── export_edge_model.py                    # Export to ONNX/TFLite/TensorRT
│   ├── edge_deployer.py                        # Complete deployment package
│   └── infer_edge.py                           # Edge inference script
│
├── 📂 Utilities
│   ├── compare_models_v2.py                    # Side-by-side model comparison
│   ├── rank_all_models.py                      # Rank all trained models
│   ├── monitor_training.py                     # Real-time training monitor
│   ├── validate_model.py                       # Model validation utility
│   └── cleanup_val_files.sh                    # Cleanup validation artifacts
│
├── 📂 Dataset
│   ├── dataset_root/                           # Main dataset (2,950 images)
│   │   ├── train/ (2,948 images)
│   │   ├── val/ (169 images)
│   │   └── test/ (149 images)
│   └── data.yaml                               # Dataset configuration
│
├── 📂 Results & Documentation
│   ├── definitive_results.txt                  # Final results summary
│   ├── final_ensemble_results.txt              # Ensemble performance
│   ├── all_models_ranked.txt                   # Complete model rankings
│   └── strategy_comparison.json                # Confidence threshold sweep
│
├── 📂 Model Weights
│   ├── yolo11n.pt, yolo11s.pt, yolo11m.pt     # Pretrained weights
│   └── runs/detect/                            # Trained model checkpoints
│
└── 📂 depricated/                              # Archived experiments & failed approaches
    ├── Failed preprocessing (LAB, LCH, dehazing)
    ├── Old training scripts
    └── Experimental hyperparameter configs
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/underwater-fish-detection.git
cd underwater-fish-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
yolo checks
```

### 2. Dataset Setup

Your dataset should follow this structure:
```
dataset_root/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 3. Training a Single Model

```bash
# Train with best configuration (ultra-stable)
python train_ultra_stable.py \
    --data dataset_root/data.yaml \
    --epochs 150 \
    --batch 64 \
    --name my_fish_detector

# Monitor training in real-time
python monitor_training.py --run my_fish_detector
```

### 4. Running the Ensemble

```bash
# Run 5-model ensemble inference
python true_ensemble_inference.py \
    --models \
        runs/detect/extreme_stable_v1/weights/best.pt \
        runs/detect/exp3_lab_aggressive_conservative/weights/best.pt \
        runs/detect/ultra_stable_v1/weights/best.pt \
        runs/detect/small_precision_v2_scratch/weights/best.pt \
        runs/detect/ultra_stable_from_coco_scratch/weights/best.pt \
    --conf 0.45 \
    --source test_images/
```

### 5. Labeling New Data

Our ensemble-based labeling system supports 3 modes:

```bash
# Mode 1: Standard YOLO format labels
python label_dataset_with_ensemble.py \
    --mode standard \
    --input-dir unlabeled_images \
    --output-dir labeled_dataset

# Mode 2: Visualized bounding boxes
python label_dataset_with_ensemble.py \
    --mode visualized \
    --input-dir sample_images \
    --output-dir visualized_output

# Mode 3: Live video labeling
python label_dataset_with_ensemble.py \
    --mode video \
    --video-path underwater_footage.mp4 \
    --output-path labeled_video.mp4
```

### 6. Edge Deployment

```bash
# Export to ONNX (FP16)
python export_edge_model.py \
    --weights runs/detect/my_fish_detector/weights/best.pt \
    --format onnx \
    --half

# Export to TFLite (INT8) for smallest size
python export_edge_model.py \
    --weights runs/detect/my_fish_detector/weights/best.pt \
    --format tflite \
    --int8
```

---

## 🔬 Methodology

### Dataset Characteristics

- **Total Images**: 2,950 (Train: 2,948, Val: 169, Test: 149)
- **Resolution**: 768×432 pixels (native underwater camera aspect ratio)
- **Source**: Extracted from underwater video at 1 FPS
- **Preprocessing**: Roboflow pipeline (auto-contrast, brightness jitter ±25%, noise 1%)

**Visual Challenges**:
- Extreme blue tint (mean blue value: 122/255)
- Low contrast (<1.5:1 in many regions)
- Water turbidity (visibility <5 meters)
- Severe class imbalance (8.73:1 ratio)

### Training Strategy

Through 60+ experiments, we discovered the **ultra-stable training paradigm**:

```yaml
# hyp_fish_extreme_stable.yaml
lr0: 0.00005          # Very low learning rate
lrf: 0.01             # Low final LR
batch: 64-80          # Large batch sizes
momentum: 0.937       # Standard
weight_decay: 0.0005
warmup_epochs: 3.0
optimizer: 'AdamW'    # Better than SGD for small datasets
cos_lr: True          # Cosine LR scheduling
```

**Key Findings**:
- ✅ **Ultra-conservative hyperparameters outperformed aggressive tuning**
- ✅ **Roboflow's simple preprocessing beat all custom approaches** (LAB, LCH, dehazing all failed)
- ✅ **Large batches (64-80) dramatically improved stability**
- ✅ **Single-model ceiling at 65-67%** due to data quality limits

### Ensemble Configuration

**5-Model Ensemble** (Sequential Execution):
1. `exp3_lab_aggressive_conservative` - Best balanced (68.87% prec, 62.58% rec)
2. `ultra_stable_v1` - High recall (62.60% prec, 68.60% rec)
3. `extreme_stable_v1` - Balanced (63.73% prec, 66.15% rec)
4. `small_precision_v2_scratch` - High precision (74.64% prec, 57.02% rec)
5. `ultra_stable_from_coco_scratch` - Extreme precision (74.51% prec, 50.78% rec)

**Optimization**: Confidence threshold sweep (0.25, 0.35, 0.45, 0.55) → **0.45 optimal**

---

## 📈 Experimental Results

### What Worked ✅

| Approach | Result | Improvement |
|----------|--------|-------------|
| Ultra-stable training (lr=0.00005) | 65.28% | Best single model |
| Large batch sizes (64-80) | Stable convergence | Reduced noise |
| Roboflow default preprocessing | 65.28% baseline | Simple works best |
| Cosine LR scheduling | Consistent in top-10 | Better convergence |
| 5-model ensemble | 70.11% | +4.38% improvement |
| Confidence optimization (0.45) | 70.11% | +0.85% over baseline |

### What Did NOT Work ❌

| Approach | Result | vs Baseline |
|----------|--------|-------------|
| LAB color correction | 45.17% | **-20.11%** |
| LCH hue shifting | 45.17% | **-20.11%** |
| Extreme RGB manipulation | 49.58% | **-15.70%** |
| Dark channel dehazing | 51.44% | **-13.84%** |
| CLAHE enhancement | 52.36% | **-12.92%** |

**Critical Lesson**: Custom preprocessing for underwater images destroyed subtle fish textures. Simple preprocessing (Roboflow defaults) preserved essential features.

---

## 🎓 Academic Contributions

### Novel Findings

1. **Data Quality Ceiling**: Identified fundamental 65-67% accuracy limit for single models on degraded underwater imagery
2. **Ultra-Stable Training Paradigm**: Demonstrated that extremely conservative hyperparameters (lr=0.00005, batch=80) outperform standard YOLO training
3. **Preprocessing Failure Analysis**: Comprehensive documentation of why underwater-specific preprocessing (LAB, dehazing) failed catastrophically
4. **Ensemble as Ceiling-Breaker**: Proved ensemble methods can overcome data quality limitations (+4.38% improvement)

### Reproducible Research

- ✅ All 60+ experiments documented with hyperparameters
- ✅ Failed approaches thoroughly analyzed (not just successes)
- ✅ Complete codebase with training/inference/deployment scripts
- ✅ Systematic methodology applicable to other low-resource CV tasks

---

## 📚 Documentation

### Main Reports
- **[Complete Project Report](FISH_DETECTION_PROJECT_REPORT.md)** - 1,900+ line technical report covering:
  - Dataset analysis and challenges
  - 60+ experiment results and analysis
  - Design decisions and rationale
  - Failure mode analysis
  - Ensemble methodology
  - Future work recommendations

### Guides
- **[Dataset Labeling Guide](DATASET_LABELING_GUIDE.md)** - Complete instructions for 3-mode labeling system

### Results Files
- `definitive_results.txt` - Final model rankings
- `final_ensemble_results.txt` - Ensemble configuration and performance
- `strategy_comparison.json` - Confidence threshold sweep data
- `all_models_ranked.txt` - Complete model comparison

---

## 🔧 Advanced Usage

### Model Comparison

Compare multiple trained models side-by-side:
```bash
python compare_models_v2.py \
    --models \
        runs/detect/model1/weights/best.pt \
        runs/detect/model2/weights/best.pt \
    --test-data dataset_root/test
```

### Hyperparameter Tuning

Experiment with different configurations:
```bash
# Precision-focused training
python train_ultra_stable_v2.py \
    --data dataset_root/data.yaml \
    --hyp hyp_precision_focus_v2.yaml \
    --epochs 150 \
    --name precision_model

# Recall-focused training
python train_ultra_stable_v2.py \
    --data dataset_root/data.yaml \
    --hyp hyp_recall_focus_v2.yaml \
    --epochs 150 \
    --name recall_model
```

### Batch Inference

Process entire directories:
```bash
python infer_edge.py \
    --model best.pt \
    --source test_images/ \
    --save-json \
    --conf 0.45
```

---

## 🎯 Performance Targets

### For Production Deployment

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Average Accuracy | >70% | 70.11% | ✅ PASS |
| Model Size (per model) | <70 MB | 18.3 MB | ✅ PASS |
| Precision | >65% | 66.27% | ✅ PASS |
| Recall | >65% | 73.94% | ✅ PASS |
| F1 Score | >65% | 69.89% | ✅ PASS |
| Edge Deployment | Yes | Yes | ✅ PASS |

**All production targets exceeded!** ✅

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Training accuracy plateaus at ~60-65%
```bash
# This is the single-model ceiling due to data quality
# Solution: Use ensemble methods
python true_ensemble_inference.py --models [list of models]
```

**Issue**: High precision, low recall
```bash
# Model is too conservative
# Solution: Lower confidence threshold
--conf 0.25  # or even 0.15
```

**Issue**: Model size exceeds 70MB
```bash
# Use INT8 quantization
python export_edge_model.py \
    --weights best.pt \
    --format tflite \
    --int8
# Result: ~5-6 MB model
```

**Issue**: Slow inference on edge device
```bash
# Use ONNX FP16 or TFLite INT8
python export_edge_model.py \
    --weights best.pt \
    --format onnx \
    --half
```

---

## 📊 Experiment Tracking

This project used [Weights & Biases](https://wandb.ai/) for experiment tracking:

- 13-18 metrics tracked per epoch
- Real-time training curves
- Automatic checkpoint saving
- Hyperparameter comparison
- 60+ experiments systematically logged

To enable W&B tracking:
```bash
wandb login
# Then run any training script - W&B will auto-log
```

---

## 🚢 Deployment Guide

### Edge Device Requirements

**Recommended Hardware**:
- NVIDIA Jetson Nano (4GB): 10-15 FPS
- NVIDIA Jetson Xavier NX (8GB): 20-30 FPS
- Raspberry Pi 4 (with Coral TPU): 15-20 FPS
- Mobile devices (with NNAPI): 10-15 FPS

### Deployment Steps

1. **Export to appropriate format**:
   ```bash
   python export_edge_model.py \
       --weights best.pt \
       --format onnx \
       --half
   ```

2. **Test inference speed**:
   ```bash
   python infer_edge.py \
       --model best.onnx \
       --source test.jpg \
       --benchmark
   ```

3. **Deploy to device**:
   ```bash
   python edge_deployer.py \
       --model best.onnx \
       --deploy-to jetson
   ```

---

## 🔮 Future Work

### To Achieve 75%+ Accuracy

1. **Expand Dataset**:
   - Collect 3,000+ Parrot Fish instances (vs. current 564)
   - Total 10,000+ images (vs. current 2,950)
   - Improve image quality (clear water, better lighting)

2. **Advanced Architectures**:
   - YOLOv11m with quantization (if size limit relaxed to 100MB)
   - Attention mechanisms for small object detection
   - Multi-scale feature fusion

3. **Active Learning**:
   - Use ensemble to label new data
   - Human-in-the-loop validation
   - Iterative dataset expansion

4. **Domain Adaptation**:
   - Transfer learning from larger marine datasets
   - Synthetic data generation
   - Cross-domain augmentation

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{underwater_fish_yolov11_2024,
  title={Underwater Fish Detection with YOLOv11: Achieving 70% Accuracy Through Ensemble Methods},
  author={[Amaan Mansuri, Vishwa Raval and Shravan Khunti]},
  year={2025},
  note={Capstone Project - [New York University | BlueBOT | UNDP]},
  howpublished={\url{https://github.com/VishwaRaval/cap_11}}
}
```

---

## 🤝 Contributing

This project is complete as a capstone submission, but improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Author

**[Amaan Mansuri, Vishwa Raval and Shravan Khunti]**
- University: [New York University]
- Program: [Masters in Data Science]
- Year: 2025
- Capstone Project

---

## 🙏 Acknowledgments

- **Ultralytics** for the excellent YOLOv11 framework
- **Roboflow** for dataset preprocessing tools
- **Weights & Biases** for experiment tracking
- **Advisors**: [Antonio Hollingsworth] for guidance and feedback
- **Dataset**: [Dataset source/provider if applicable]

---


## ⚡ Quick Links

- [📄 Complete Technical Report](FISH_DETECTION_PROJECT_REPORT.md)
- [📖 Dataset Labeling Guide](DATASET_LABELING_GUIDE.md)
- [🔧 Ultralytics Docs](https://docs.ultralytics.com)

---

<div align="center">

**🐠 Successfully Detecting Fish in Challenging Underwater Environments 🐠**

*A systematic approach to low-resource computer vision*

</div>
