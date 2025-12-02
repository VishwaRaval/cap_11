#!/usr/bin/env python3
"""
Specialized Underwater Preprocessing for SEVERE Blue Tint
Combines multiple techniques to handle extreme underwater conditions

Usage:
    python preprocess_underwater_extreme.py \
        --input dataset_root \
        --output dataset_root_extreme_corrected
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


def white_balance_underwater(image):
    """
    Specialized white balance for underwater images with severe blue tint.
    Uses gray world assumption with channel-specific correction.
    """
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Calculate average for each channel
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    
    # Apply correction (compensate for blue shift)
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.2)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.2)
    
    result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


def reduce_blue_channel_dominance(image, reduction_factor=0.7):
    """
    Directly reduce blue channel dominance.
    Aggressive approach for severe blue tint.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    
    # Calculate how much blue dominates
    blue_excess = np.maximum(b - (r + g) / 2, 0)
    
    # Reduce blue channel where it's excessive
    b_corrected = b - (blue_excess * reduction_factor)
    
    # Slightly boost red channel (opposite of blue)
    r_boosted = r + (blue_excess * reduction_factor * 0.3)
    
    # Clip to valid range
    b_corrected = np.clip(b_corrected, 0, 255)
    r_boosted = np.clip(r_boosted, 0, 255)
    
    corrected = cv2.merge([b_corrected, g, r_boosted]).astype(np.uint8)
    return corrected


def adaptive_clahe(image, clip_limit=4.0, tile_size=8):
    """
    CLAHE with adaptive parameters based on image characteristics.
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    
    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return result


def enhance_underwater_visibility(image):
    """
    Multi-stage enhancement specifically for underwater images.
    """
    # Stage 1: White balance correction
    wb_corrected = white_balance_underwater(image)
    
    # Stage 2: Reduce blue dominance
    blue_corrected = reduce_blue_channel_dominance(wb_corrected, reduction_factor=0.7)
    
    # Stage 3: CLAHE for local contrast
    clahe_enhanced = adaptive_clahe(blue_corrected, clip_limit=4.0, tile_size=8)
    
    # Stage 4: Slight sharpening
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]]) * 0.2
    sharpened = cv2.filter2D(clahe_enhanced, -1, kernel)
    
    # Blend sharpened with clahe_enhanced (50-50)
    result = cv2.addWeighted(clahe_enhanced, 0.7, sharpened, 0.3, 0)
    
    return result


def process_image(input_path, output_path):
    """
    Process a single image with extreme underwater correction.
    """
    # Read image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Warning: Could not read {input_path}")
        return False
    
    # Apply enhancement
    enhanced = enhance_underwater_visibility(img)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), enhanced)
    return True


def preprocess_dataset(input_root, output_root):
    """
    Preprocess entire dataset with extreme underwater correction.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    # Create output root
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Copy data.yaml and other files
    for file in input_root.glob("*.yaml"):
        shutil.copy2(file, output_root / file.name)
    for file in input_root.glob("*.txt"):
        shutil.copy2(file, output_root / file.name)
    
    # Process each split
    splits = ['train', 'valid', 'test']
    
    total_processed = 0
    total_failed = 0
    
    for split in splits:
        split_dir = input_root / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        # Get images and labels
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        print(f"\n{'='*70}")
        print(f"Processing {split} split: {len(image_files)} images")
        print(f"{'='*70}")
        
        # Create output directories
        output_images_dir = output_root / split / 'images'
        output_labels_dir = output_root / split / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        for img_path in tqdm(image_files, desc=f"{split} images"):
            output_img_path = output_images_dir / img_path.name
            
            success = process_image(img_path, output_img_path)
            if success:
                total_processed += 1
            else:
                total_failed += 1
            
            # Copy corresponding label file
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                output_label_path = output_labels_dir / label_path.name
                shutil.copy2(label_path, output_label_path)
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {total_processed} images")
    if total_failed > 0:
        print(f"✗ Failed: {total_failed} images")
    print(f"✓ Output saved to: {output_root}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extreme underwater preprocessing for severe blue tint"
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input dataset root directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for preprocessed dataset')
    
    args = parser.parse_args()
    
    print("="*70)
    print("EXTREME UNDERWATER PREPROCESSING")
    print("="*70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("\nEnhancements applied:")
    print("  1. White balance correction (gray world)")
    print("  2. Blue channel reduction (70% reduction)")
    print("  3. Red channel boost (compensate for blue)")
    print("  4. Adaptive CLAHE (clip=4.0)")
    print("  5. Sharpening (enhance edges)")
    print("="*70 + "\n")
    
    # Run preprocessing
    preprocess_dataset(args.input, args.output)
    
    print("\n💡 NEXT STEPS:")
    print("-"*70)
    print("1. Check quality: python check_image_quality.py --dataset", args.output)
    print("2. If blue tint < 50: Proceed with training")
    print("3. If still > 50: Dataset may need manual color grading")
    print("-"*70)


if __name__ == '__main__':
    main()
