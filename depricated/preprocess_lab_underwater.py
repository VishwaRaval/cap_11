#!/usr/bin/env python3
"""
LAB Color Space Preprocessing for Underwater Fish Images
Conservative blue tint correction targeting the B channel

Usage:
    python preprocess_lab_underwater.py \
        --input /path/to/original_dataset \
        --output /path/to/lab_corrected_dataset \
        --blue-reduction 0.7 \
        --visualize 10
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_blue_tint(image):
    """
    Calculate blue tint metric for an image.
    Returns mean blue channel value (higher = more blue cast).
    """
    if len(image.shape) == 3:
        # BGR image
        blue_channel = image[:, :, 0]
        return np.mean(blue_channel)
    return 0.0


def lab_underwater_correction(image, blue_reduction=0.7, gamma=1.0):
    """
    Conservative LAB color correction for underwater images.
    
    Args:
        image: BGR image (OpenCV format)
        blue_reduction: How much to reduce blue cast (0.5-0.9)
                       0.7 = reduce blue tint by 30%
                       Higher values = more aggressive correction
        gamma: Optional gamma correction for brightness (1.0 = no change)
    
    Returns:
        corrected_image: BGR image with reduced blue cast
    """
    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # --- L channel (Lightness) ---
    # Optional: Apply gamma correction if image is too dark
    if gamma != 1.0:
        l = l.astype(np.float32) / 255.0
        l = np.power(l, gamma)
        l = (l * 255.0).astype(np.uint8)
    
    # --- A channel (Green-Red) ---
    # Leave A channel unchanged - preserves natural color balance
    
    # --- B channel (Yellow-Blue) ---
    # This is where underwater blue cast lives
    # B channel: 0 = blue, 128 = neutral, 255 = yellow
    
    # Calculate how far we are from neutral (128)
    b_mean = np.mean(b)
    b_shift = (128 - b_mean) * (1 - blue_reduction)
    
    # Apply shift toward yellow (reduce blue) - conservatively
    b_corrected = b.astype(np.float32) + b_shift
    b_corrected = np.clip(b_corrected, 0, 255).astype(np.uint8)
    
    # Optional: Slight contrast enhancement in B channel
    # This helps if the correction makes colors too flat
    # b_corrected = cv2.equalizeHist(b_corrected)  # Uncomment if needed
    
    # Merge corrected channels
    lab_corrected = cv2.merge([l, a, b_corrected])
    
    # Convert back to BGR
    corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
    
    return corrected


def process_dataset(input_root, output_root, blue_reduction=0.7, gamma=1.0, 
                   visualize_n=0, quality_threshold=50):
    """
    Process entire dataset with LAB correction.
    
    Args:
        input_root: Path to original dataset (YOLOv11 format)
        output_root: Path to output corrected dataset
        blue_reduction: Blue tint reduction factor
        gamma: Gamma correction factor
        visualize_n: Number of before/after samples to save
        quality_threshold: Max acceptable blue tint value (0-255)
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    # Create output directory structure
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total_images': 0,
        'processed': 0,
        'before_blue_tint': [],
        'after_blue_tint': [],
        'improved': 0,
        'failed': 0,
    }
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        input_images = input_root / split / 'images'
        input_labels = input_root / split / 'labels'
        
        if not input_images.exists():
            print(f"⚠️  Skipping {split} - not found")
            continue
        
        output_images = output_root / split / 'images'
        output_labels = output_root / split / 'labels'
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(input_images.glob('*.jpg')) + list(input_images.glob('*.png'))
        
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split: {len(image_files)} images")
        print(f"{'='*70}")
        
        viz_count = 0
        viz_dir = output_root / 'visualizations' / split
        if visualize_n > 0:
            viz_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(image_files, desc=f"{split}"):
            stats['total_images'] += 1
            
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"❌ Failed to read: {img_path.name}")
                stats['failed'] += 1
                continue
            
            # Calculate before metrics
            blue_before = calculate_blue_tint(image)
            stats['before_blue_tint'].append(blue_before)
            
            # Apply LAB correction
            corrected = lab_underwater_correction(
                image, 
                blue_reduction=blue_reduction,
                gamma=gamma
            )
            
            # Calculate after metrics
            blue_after = calculate_blue_tint(corrected)
            stats['after_blue_tint'].append(blue_after)
            
            # Check if improved
            if blue_after < blue_before and blue_after <= quality_threshold:
                stats['improved'] += 1
            
            # Save corrected image
            output_path = output_images / img_path.name
            cv2.imwrite(str(output_path), corrected)
            
            # Copy corresponding label file
            label_path = input_labels / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, output_labels / label_path.name)
            
            stats['processed'] += 1
            
            # Visualization
            if visualize_n > 0 and viz_count < visualize_n:
                viz_count += 1
                save_comparison(image, corrected, blue_before, blue_after, 
                              viz_dir / f"{img_path.stem}_comparison.png")
    
    # Copy data.yaml
    data_yaml = input_root / 'data.yaml'
    if data_yaml.exists():
        shutil.copy2(data_yaml, output_root / 'data.yaml')
        print(f"\n✅ Copied data.yaml")
    
    # Print statistics
    print_statistics(stats, output_root, blue_reduction, gamma, quality_threshold)
    
    return stats


def save_comparison(original, corrected, blue_before, blue_after, output_path):
    """Save before/after comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(original_rgb)
    axes[0].set_title(f'Original\nBlue Tint: {blue_before:.1f}')
    axes[0].axis('off')
    
    axes[1].imshow(corrected_rgb)
    axes[1].set_title(f'LAB Corrected\nBlue Tint: {blue_after:.1f}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_statistics(stats, output_root, blue_reduction, gamma, quality_threshold):
    """Print processing statistics."""
    print(f"\n{'='*70}")
    print("LAB PREPROCESSING STATISTICS")
    print(f"{'='*70}")
    
    print(f"\n📊 Processing Summary:")
    print(f"  Total images:        {stats['total_images']}")
    print(f"  Successfully processed: {stats['processed']}")
    print(f"  Failed:              {stats['failed']}")
    
    if stats['before_blue_tint']:
        mean_before = np.mean(stats['before_blue_tint'])
        mean_after = np.mean(stats['after_blue_tint'])
        reduction = ((mean_before - mean_after) / mean_before) * 100
        
        print(f"\n🎨 Color Correction Results:")
        print(f"  Blue tint BEFORE:    {mean_before:.1f} (mean)")
        print(f"  Blue tint AFTER:     {mean_after:.1f} (mean)")
        print(f"  Reduction:           {reduction:.1f}%")
        
        # Quality assessment
        images_below_threshold = sum(1 for b in stats['after_blue_tint'] if b <= quality_threshold)
        pct_good = (images_below_threshold / len(stats['after_blue_tint'])) * 100
        
        print(f"\n✅ Quality Assessment:")
        print(f"  Images with blue < {quality_threshold}: {images_below_threshold}/{len(stats['after_blue_tint'])} ({pct_good:.1f}%)")
        print(f"  Images improved:     {stats['improved']} ({stats['improved']/stats['processed']*100:.1f}%)")
    
    print(f"\n⚙️  Parameters Used:")
    print(f"  Blue reduction:      {blue_reduction} ({(1-blue_reduction)*100:.0f}% correction)")
    print(f"  Gamma:               {gamma}")
    print(f"  Quality threshold:   {quality_threshold}")
    
    print(f"\n📁 Output Location:")
    print(f"  {output_root}")
    
    print(f"{'='*70}\n")
    
    # Save statistics to file
    stats_file = output_root / 'preprocessing_stats.txt'
    with open(stats_file, 'w') as f:
        f.write(f"LAB Preprocessing Statistics\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Total images: {stats['total_images']}\n")
        f.write(f"Processed: {stats['processed']}\n")
        f.write(f"Failed: {stats['failed']}\n\n")
        
        if stats['before_blue_tint']:
            f.write(f"Blue tint before: {mean_before:.1f}\n")
            f.write(f"Blue tint after: {mean_after:.1f}\n")
            f.write(f"Reduction: {reduction:.1f}%\n\n")
            f.write(f"Images below threshold ({quality_threshold}): {pct_good:.1f}%\n")
            f.write(f"Images improved: {stats['improved']/stats['processed']*100:.1f}%\n\n")
        
        f.write(f"Parameters:\n")
        f.write(f"  blue_reduction: {blue_reduction}\n")
        f.write(f"  gamma: {gamma}\n")
    
    print(f"✅ Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="LAB Color Space Preprocessing for Underwater Images"
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input dataset (YOLOv11 format)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output corrected dataset')
    parser.add_argument('--blue-reduction', type=float, default=0.7,
                       help='Blue tint reduction factor (0.5-0.9, default: 0.7)')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Gamma correction for brightness (default: 1.0 = no change)')
    parser.add_argument('--visualize', type=int, default=10,
                       help='Number of before/after samples to visualize per split (default: 10)')
    parser.add_argument('--quality-threshold', type=int, default=50,
                       help='Max acceptable blue tint value (0-255, default: 50)')
    
    args = parser.parse_args()
    
    # Validate parameters
    if not (0.3 <= args.blue_reduction <= 0.95):
        print("⚠️  Warning: blue_reduction should be between 0.3-0.95")
        print(f"   Using: {args.blue_reduction}")
    
    if not (0.5 <= args.gamma <= 2.0):
        print("⚠️  Warning: gamma should be between 0.5-2.0")
        print(f"   Using: {args.gamma}")
    
    print("\n" + "="*70)
    print("LAB COLOR SPACE PREPROCESSING")
    print("="*70)
    print(f"📂 Input:  {args.input}")
    print(f"📂 Output: {args.output}")
    print(f"🎨 Blue reduction: {args.blue_reduction} ({(1-args.blue_reduction)*100:.0f}% correction)")
    print(f"💡 Gamma: {args.gamma}")
    print(f"📊 Quality threshold: {args.quality_threshold}")
    print(f"🖼️  Visualizations: {args.visualize} per split")
    print("="*70 + "\n")
    
    # Process dataset
    stats = process_dataset(
        input_root=args.input,
        output_root=args.output,
        blue_reduction=args.blue_reduction,
        gamma=args.gamma,
        visualize_n=args.visualize,
        quality_threshold=args.quality_threshold
    )
    
    print("\n✅ LAB preprocessing complete!")
    print(f"📁 Corrected dataset ready at: {args.output}")
    print(f"🖼️  Visualizations saved in: {args.output}/visualizations/")
    print("\n🚀 Next step: Train YOLOv11 on the corrected dataset")
    print(f"\nExample command:")
    print(f"python train_yolo11_fish_enhanced_fixed.py \\")
    print(f"    --data {args.output} \\")
    print(f"    --model runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt \\")
    print(f"    --hyp hyp_fish_recall_optimized.yaml \\")
    print(f"    --epochs 75 \\")
    print(f"    --batch 16 \\")
    print(f"    --use-class-weights \\")
    print(f"    --name lab_corrected_finetune_v1\n")


if __name__ == '__main__':
    main()
