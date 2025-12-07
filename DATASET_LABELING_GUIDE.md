# Dataset Labeling Guide with Ensemble Model

This guide provides step-by-step instructions for labeling your entire underwater fish dataset using the trained ensemble model in three different ways.

## Prerequisites

```bash
# Ensure you have the required packages
pip install ultralytics opencv-python tqdm numpy

# Verify your ensemble models are available
ls runs/detect/extreme_stable_v1/weights/best.pt
ls runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt
ls runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt
ls runs/detect/fish_s_s_cosine_ultra_v1/weights/best.pt
```

## Model Configuration

Your ensemble uses:
- **Models**: 4 YOLOv11s models with diverse training configurations
- **Weights**: [0.40, 0.30, 0.15, 0.15] (optimized weighted averaging)
- **Confidence threshold**: 0.25
- **IoU threshold**: 0.45
- **Test-Time Augmentation**: Horizontal flip + brightness jitter (for higher accuracy)

---

## Mode 1: Standard YOLO Format Labels

**Use case**: Creating a new labeled dataset for further training or data augmentation.

### Output Structure
```
output_dir/
├── images/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
├── labels/
│   ├── frame_001.txt
│   ├── frame_002.txt
│   └── ...
├── dataset.yaml
└── labeling_stats.json
```

### Label Format
Each `.txt` file contains YOLO format annotations:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized (0-1).

### Step-by-Step Instructions

#### Step 1: Prepare your input images
```bash
# Create a directory with your unlabeled images
mkdir -p unlabeled_images
# Copy or move your images there
cp /path/to/video_frames/*.jpg unlabeled_images/
```

#### Step 2: Run the labeling script
```bash
python label_dataset_with_ensemble.py \
    --mode standard \
    --input-dir unlabeled_images \
    --output-dir labeled_dataset \
    --models \
        runs/detect/extreme_stable_v1/weights/best.pt \
        runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt \
        runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt \
        runs/detect/fish_s_s_cosine_ultra_v1/weights/best.pt \
    --weights 0.40 0.30 0.15 0.15 \
    --conf 0.25 \
    --iou 0.45
```

#### Step 3: Verify the output
```bash
# Check the generated files
ls labeled_dataset/images/ | wc -l  # Should match input count
ls labeled_dataset/labels/ | wc -l  # Should match input count

# View statistics
cat labeled_dataset/labeling_stats.json
```

#### Step 4: Split dataset for training (optional)
```bash
# Create train/val/test splits
python -c "
import os
import shutil
from pathlib import Path
import random

random.seed(42)

src = Path('labeled_dataset')
images = list((src / 'images').glob('*.jpg'))
random.shuffle(images)

# 80/10/10 split
train_split = int(0.8 * len(images))
val_split = int(0.9 * len(images))

splits = {
    'train': images[:train_split],
    'val': images[train_split:val_split],
    'test': images[val_split:]
}

for split_name, split_images in splits.items():
    # Create directories
    (src / split_name / 'images').mkdir(parents=True, exist_ok=True)
    (src / split_name / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for img_path in split_images:
        shutil.copy(img_path, src / split_name / 'images' / img_path.name)
        label_path = src / 'labels' / f'{img_path.stem}.txt'
        if label_path.exists():
            shutil.copy(label_path, src / split_name / 'labels' / label_path.name)

print(f'Train: {len(splits[\"train\"])}')
print(f'Val: {len(splits[\"val\"])}')
print(f'Test: {len(splits[\"test\"])}')
"
```

#### Step 5: Update dataset.yaml
```yaml
# labeled_dataset/dataset.yaml
path: /absolute/path/to/labeled_dataset
train: train/images
val: val/images
test: test/images

nc: 3
names: ['Grunt Fish', 'Parrot Fish', 'Surgeon Fish']
```

### Expected Output
```
✅ Labeling complete!
   Images: labeled_dataset/images
   Labels: labeled_dataset/labels
   Total detections: 8547
   - Grunt Fish: 2156
   - Parrot Fish: 612
   - Surgeon Fish: 5779
```

---

## Mode 2: Images with Visualized Bounding Boxes

**Use case**: Visual inspection, presentations, debugging, or creating annotated datasets for review.

### Output
Each image is saved with:
- Colored bounding boxes (class-specific colors)
- Class labels
- Confidence scores

### Step-by-Step Instructions

#### Step 1: Prepare input directory
```bash
# Use the same unlabeled images or a subset
mkdir -p images_to_visualize
cp unlabeled_images/*.jpg images_to_visualize/
```

#### Step 2: Run visualization
```bash
python label_dataset_with_ensemble.py \
    --mode visualized \
    --input-dir images_to_visualize \
    --output-dir visualized_output \
    --models \
        runs/detect/extreme_stable_v1/weights/best.pt \
        runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt \
        runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt \
        runs/detect/fish_s_s_cosine_ultra_v1/weights/best.pt \
    --weights 0.40 0.30 0.15 0.15
```

#### Step 3: Hide confidence scores (optional)
```bash
# If you don't want confidence scores shown
python label_dataset_with_ensemble.py \
    --mode visualized \
    --input-dir images_to_visualize \
    --output-dir visualized_output_clean \
    --no-confidence
```

#### Step 4: Review results
```bash
# Open a few images to verify
eog visualized_output/frame_001_labeled.jpg
# Or use any image viewer
```

### Color Coding
- **Grunt Fish**: Green boxes
- **Parrot Fish**: Blue boxes
- **Surgeon Fish**: Orange boxes

### Creating a Comparison Grid (Optional)
```bash
# Create side-by-side comparison
python -c "
import cv2
import numpy as np
from pathlib import Path

input_dir = Path('images_to_visualize')
output_dir = Path('visualized_output')
comparison_dir = Path('comparison_grid')
comparison_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob('*.jpg'):
    # Load original
    original = cv2.imread(str(img_path))
    
    # Load labeled
    labeled_path = output_dir / f'{img_path.stem}_labeled.jpg'
    if labeled_path.exists():
        labeled = cv2.imread(str(labeled_path))
        
        # Resize if needed
        h, w = original.shape[:2]
        labeled = cv2.resize(labeled, (w, h))
        
        # Create side-by-side
        combined = np.hstack([original, labeled])
        
        # Save
        cv2.imwrite(str(comparison_dir / f'{img_path.stem}_comparison.jpg'), combined)

print('✓ Comparison grid created')
"
```

---

## Mode 3: Live Video Labeling

**Use case**: Processing video footage with frame-by-frame predictions, creating demo videos, or analyzing temporal patterns.

### Output
A video file with:
- Real-time bounding box predictions
- Class labels and confidence scores
- Frame counter and detection count overlay

### Step-by-Step Instructions

#### Step 1: Prepare video file
```bash
# Ensure your video is accessible
ls underwater_footage.mp4
# Or extract video from dataset if needed
```

#### Step 2: Run video labeling
```bash
python label_dataset_with_ensemble.py \
    --mode video \
    --video-path underwater_footage.mp4 \
    --output-path labeled_video.mp4 \
    --models \
        runs/detect/extreme_stable_v1/weights/best.pt \
        runs/detect/fish_m_m_recall_optimized_v1/weights/best.pt \
        runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt \
        runs/detect/fish_s_s_cosine_ultra_v1/weights/best.pt \
    --weights 0.40 0.30 0.15 0.15
```

#### Step 3: Adjust output FPS (optional)
```bash
# Reduce FPS for smaller file size
python label_dataset_with_ensemble.py \
    --mode video \
    --video-path underwater_footage.mp4 \
    --output-path labeled_video_15fps.mp4 \
    --fps 15
```

#### Step 4: Disable TTA for faster processing
```bash
# For very long videos, disable TTA to speed up
python label_dataset_with_ensemble.py \
    --mode video \
    --video-path long_footage.mp4 \
    --output-path labeled_long_video.mp4 \
    --no-tta
```

### Performance Notes
- **With TTA**: ~2-3 FPS on A100 (highest accuracy)
- **Without TTA**: ~8-10 FPS on A100 (faster, slightly lower accuracy)
- **On Jetson Nano**: Expect 5-8 FPS without TTA

### Processing Large Videos
For videos longer than 10 minutes:

```bash
# Option 1: Split video into chunks
ffmpeg -i long_video.mp4 -c copy -map 0 -segment_time 00:05:00 -f segment output%03d.mp4

# Process each chunk
for file in output*.mp4; do
    python label_dataset_with_ensemble.py \
        --mode video \
        --video-path $file \
        --output-path labeled_$file \
        --no-tta
done

# Merge labeled chunks
ffmpeg -f concat -safe 0 -i <(for f in labeled_output*.mp4; do echo "file '$PWD/$f'"; done) -c copy final_labeled_video.mp4
```

---

## Advanced Usage

### Custom Model Paths
If your models are in different locations:

```bash
python label_dataset_with_ensemble.py \
    --mode standard \
    --input-dir unlabeled_images \
    --output-dir labeled_dataset \
    --models \
        /path/to/model1.pt \
        /path/to/model2.pt \
        /path/to/model3.pt \
        /path/to/model4.pt \
    --weights 0.40 0.30 0.15 0.15
```

### Adjusting Confidence Threshold
For more/fewer detections:

```bash
# More detections (lower threshold)
--conf 0.15

# Fewer, higher-confidence detections
--conf 0.35
```

### Batch Processing Multiple Directories
```bash
#!/bin/bash
# batch_label.sh

for dir in unlabeled_batch_*; do
    echo "Processing $dir..."
    python label_dataset_with_ensemble.py \
        --mode standard \
        --input-dir "$dir" \
        --output-dir "labeled_$dir"
done
```

---

## Quality Assurance

### Step 1: Sample Random Images
```bash
# Extract 50 random samples for manual review
python -c "
import random
import shutil
from pathlib import Path

labeled_dir = Path('labeled_dataset')
sample_dir = Path('qa_samples')
sample_dir.mkdir(exist_ok=True)

images = list((labeled_dir / 'images').glob('*.jpg'))
samples = random.sample(images, min(50, len(images)))

for img in samples:
    shutil.copy(img, sample_dir / img.name)
    label = labeled_dir / 'labels' / f'{img.stem}.txt'
    if label.exists():
        shutil.copy(label, sample_dir / label.name)
"
```

### Step 2: Create Visualizations of Samples
```bash
python label_dataset_with_ensemble.py \
    --mode visualized \
    --input-dir qa_samples \
    --output-dir qa_samples_visualized
```

### Step 3: Manual Review
Open the visualized samples and check:
- [ ] Bounding boxes are accurate
- [ ] Class labels are correct
- [ ] No missed fish (false negatives)
- [ ] No false detections (false positives)

### Step 4: Calculate Label Statistics
```bash
python -c "
from pathlib import Path
import json

labels_dir = Path('labeled_dataset/labels')
class_counts = {0: 0, 1: 0, 2: 0}
total_boxes = 0
empty_labels = 0

for label_file in labels_dir.glob('*.txt'):
    with open(label_file) as f:
        lines = f.readlines()
        if not lines:
            empty_labels += 1
            continue
        
        for line in lines:
            cls = int(line.split()[0])
            class_counts[cls] += 1
            total_boxes += 1

stats = {
    'total_images': len(list(labels_dir.glob('*.txt'))),
    'total_boxes': total_boxes,
    'empty_images': empty_labels,
    'class_distribution': {
        'Grunt Fish': class_counts[0],
        'Parrot Fish': class_counts[1],
        'Surgeon Fish': class_counts[2]
    }
}

print(json.dumps(stats, indent=2))
"
```

---

## Troubleshooting

### Issue: "Model not found"
**Solution**: Check model paths are correct
```bash
# List available models
ls -R runs/detect/*/weights/best.pt
```

### Issue: "Out of memory"
**Solution**: Reduce batch size or use fewer models
```bash
# Process images one at a time
# Or disable TTA with --no-tta
```

### Issue: "Video encoding failed"
**Solution**: Install correct codec
```bash
sudo apt-get install ffmpeg libavcodec-extra
```

### Issue: "Slow video processing"
**Solution**: Use --no-tta flag
```bash
python label_dataset_with_ensemble.py \
    --mode video \
    --video-path video.mp4 \
    --output-path output.mp4 \
    --no-tta  # Disables test-time augmentation
```

---

## Expected Performance

### Mode 1: Standard Labeling
- **Speed**: ~5-10 images/second (with TTA on A100)
- **Accuracy**: 70.37% (matching ensemble test performance)
- **Output size**: Same as input + small text files

### Mode 2: Visualized Labels
- **Speed**: ~5-10 images/second
- **Output size**: Same as input (images with drawn boxes)

### Mode 3: Video Labeling
- **Speed**: 2-3 FPS (with TTA), 8-10 FPS (without TTA)
- **Output size**: Similar to input video size

---

## Best Practices

1. **Always verify a sample** before labeling entire dataset
2. **Use TTA for final labeling** (higher accuracy, worth the time)
3. **Disable TTA for quick previews** or very large datasets
4. **Save statistics** for documentation and analysis
5. **Create backups** of original unlabeled data
6. **Review edge cases** (low confidence, occlusions, rare classes)

---

## Integration with Training Pipeline

Once you have labeled data:

```bash
# Train a new model with auto-labeled data
yolo detect train \
    data=labeled_dataset/dataset.yaml \
    model=yolo11s.pt \
    epochs=150 \
    imgsz=768,432 \
    batch=64 \
    name=retrain_with_autolabels
```

This creates a feedback loop:
1. Train initial model
2. Label more data with ensemble
3. Retrain with expanded dataset
4. Repeat to incrementally improve

---

## Summary

You now have three powerful ways to leverage your trained ensemble model:

1. **Standard format**: Build training datasets
2. **Visualized boxes**: Quality assurance and presentations
3. **Video labeling**: Process footage and create demos

All three modes use the same high-performing ensemble (70.37% accuracy) that exceeded your project goals.
