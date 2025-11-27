# W&B Quick Reference

## Setup (One-time)
```bash
# Install
pip install wandb

# Login with your API key
wandb login
# Or set environment variable
export WANDB_API_KEY="0a78f43170a66024d517c69952f9f8671a49b5ad"
```

## Training Commands

### Basic Training (W&B enabled by default)
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 100 \
    --batch 16
```

### Training with Custom Project Name
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 100 \
    --batch 16 \
    --wandb-project "my-fish-experiments"
```

### Training with Notes
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 100 \
    --batch 16 \
    --name baseline \
    --wandb-notes "First baseline run with preprocessed data"
```

### Training with API Key (No Login Required)
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 100 \
    --batch 16 \
    --wandb-key "0a78f43170a66024d517c69952f9f8671a49b5ad"
```

### Training WITHOUT W&B
```bash
python train_yolo11_fish.py \
    --data dataset_root_preprocessed \
    --model n \
    --epochs 100 \
    --batch 16 \
    --no-wandb
```

## What W&B Tracks

✅ Training & validation losses (per epoch)
✅ Precision, Recall, mAP metrics (per epoch)
✅ All training curves and plots
✅ Best model weights (as artifacts)
✅ Hyperparameters and configuration
✅ System metrics (GPU, CPU, memory)

## Viewing Results

Your W&B dashboard will be printed at training start:
```
✓ W&B initialized: fish_n_baseline
  Project: underwater-fish-detection
  URL: https://wandb.ai/your-username/underwater-fish-detection/runs/xxx
```

Visit that URL to see real-time training progress!

## Environment Variables (Optional)

```bash
# API Key
export WANDB_API_KEY="your_key"

# Run offline (sync later with: wandb sync)
export WANDB_MODE=offline

# Disable W&B
export WANDB_MODE=disabled

# Silent mode
export WANDB_SILENT=true
```

## Compare Multiple Runs

1. Go to your project dashboard
2. Select multiple runs (checkboxes)
3. Click "Compare" button
4. View metrics side-by-side

## Example Experiment Workflow

```bash
# Run 1: Baseline
python train_yolo11_fish.py --data dataset_root_preprocessed --model n --epochs 100 --batch 16 --name baseline

# Run 2: Enhanced augmentation
python train_yolo11_fish.py --data dataset_root_preprocessed --model n --epochs 120 --batch 16 --name enhanced_aug

# Run 3: No preprocessing
python train_yolo11_fish.py --data dataset_root --model n --epochs 100 --batch 16 --name no_preprocess

# Now compare all three in W&B dashboard!
```

## Tags (Automatically Added)

- `yolov11`
- `underwater`
- `fish-detection`
- `edge-deployment`
- `n` or `s` (model size)

## Tips

💡 Use descriptive `--name` for each run
💡 Add `--wandb-notes` to document experiment details
💡 Star your best runs in W&B UI
💡 Download model artifacts from W&B for deployment
💡 Export metrics as CSV for custom analysis

---

**Need help?** Check WANDB_GUIDE.md for detailed documentation!
