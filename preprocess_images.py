#!/usr/bin/env python3
"""
Offline preprocessing script for underwater fish images.
Applies optional dehazing and local contrast enhancement specifically for underwater scenes.

Usage:
    python preprocess_images.py --input dataset_root --output dataset_root_preprocessed
    python preprocess_images.py --input dataset_root --output dataset_root_preprocessed --dehaze --clahe --strength 1.5
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


def dark_channel_prior(image, patch_size=15):
    """
    Calculate dark channel prior for dehazing.
    Dark channel = min over RGB of min filter over local patch.
    """
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(image, dark_channel, top_percent=0.001):
    """
    Estimate atmospheric light from brightest pixels in dark channel.
    """
    h, w = dark_channel.shape
    num_pixels = int(h * w * top_percent)
    dark_flat = dark_channel.flatten()
    indices = np.argpartition(dark_flat, -num_pixels)[-num_pixels:]
    
    brightest_pixels = []
    for idx in indices:
        y, x = divmod(idx, w)
        brightest_pixels.append(image[y, x])
    
    atmospheric_light = np.max(brightest_pixels, axis=0)
    return atmospheric_light


def simple_underwater_dehaze(image, strength=1.0, omega=0.85):
    """
    Apply simple dehazing based on dark channel prior.
    Adapted for underwater scenes with conservative settings.
    
    Args:
        image: Input BGR image (uint8)
        strength: Dehazing strength multiplier (0.5-2.0 recommended)
        omega: Haze removal parameter (0.7-0.95, higher = more dehazing)
    
    Returns:
        Dehazed BGR image (uint8)
    """
    # Normalize to [0, 1]
    img_float = image.astype(np.float32) / 255.0
    
    # Calculate dark channel
    dark = dark_channel_prior(image)
    
    # Estimate atmospheric light
    A = estimate_atmospheric_light(image, dark)
    A = A.astype(np.float32) / 255.0
    
    # Estimate transmission map
    dark_norm = dark.astype(np.float32) / 255.0
    transmission = 1 - omega * dark_norm
    transmission = np.clip(transmission, 0.1, 1.0)  # Avoid division by zero
    
    # Apply strength adjustment
    transmission = np.power(transmission, 1.0 / strength)
    
    # Recover scene radiance
    transmission_3d = np.stack([transmission] * 3, axis=2)
    dehazed = np.zeros_like(img_float)
    
    for c in range(3):
        dehazed[:, :, c] = (img_float[:, :, c] - A[c]) / transmission_3d[:, :, c] + A[c]
    
    # Clip and convert back
    dehazed = np.clip(dehazed * 255, 0, 255).astype(np.uint8)
    return dehazed


def apply_clahe(image, clip_limit=2.0, tile_size=8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Better for local contrast enhancement in underwater scenes.
    
    Args:
        image: Input BGR image
        clip_limit: Contrast limiting (1.0-4.0 recommended)
        tile_size: Grid size for histogram equalization
    
    Returns:
        Enhanced BGR image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    
    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return bgr_enhanced


def process_image(input_path, output_path, apply_dehaze=False, apply_clahe_flag=False, 
                  dehaze_strength=1.0, clahe_clip=2.0):
    """
    Process a single image with optional dehazing and CLAHE.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        apply_dehaze: Whether to apply dehazing
        apply_clahe_flag: Whether to apply CLAHE
        dehaze_strength: Dehazing strength (0.5-2.0)
        clahe_clip: CLAHE clip limit (1.0-4.0)
    """
    # Read image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Warning: Could not read {input_path}")
        return False
    
    # Apply dehazing if requested
    if apply_dehaze:
        img = simple_underwater_dehaze(img, strength=dehaze_strength)
    
    # Apply CLAHE if requested
    if apply_clahe_flag:
        img = apply_clahe(img, clip_limit=clahe_clip)
    
    # Save processed image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return True


def preprocess_dataset(input_root, output_root, apply_dehaze=False, apply_clahe_flag=False,
                       dehaze_strength=1.0, clahe_clip=2.0):
    """
    Preprocess entire dataset while preserving structure.
    
    Args:
        input_root: Root directory of input dataset
        output_root: Root directory for preprocessed dataset
        apply_dehaze: Whether to apply dehazing
        apply_clahe_flag: Whether to apply CLAHE
        dehaze_strength: Dehazing strength
        clahe_clip: CLAHE clip limit
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    # Create output root
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Copy data.yaml and README files
    for file in input_root.glob("*.yaml"):
        shutil.copy2(file, output_root / file.name)
    for file in input_root.glob("*.txt"):
        shutil.copy2(file, output_root / file.name)
    
    # Process each split (train, valid, test)
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_dir = input_root / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        # Get all images
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        print(f"\nProcessing {split} split: {len(image_files)} images")
        
        # Create output directories
        output_images_dir = output_root / split / 'images'
        output_labels_dir = output_root / split / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        for img_path in tqdm(image_files, desc=f"{split} images"):
            output_img_path = output_images_dir / img_path.name
            process_image(img_path, output_img_path, apply_dehaze, apply_clahe_flag,
                         dehaze_strength, clahe_clip)
            
            # Copy corresponding label file
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                output_label_path = output_labels_dir / label_path.name
                shutil.copy2(label_path, output_label_path)
    
    print(f"\n✓ Preprocessing complete! Output saved to: {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess underwater fish dataset with optional dehazing and CLAHE"
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input dataset root directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for preprocessed dataset')
    parser.add_argument('--dehaze', action='store_true',
                       help='Apply underwater dehazing (dark channel prior)')
    parser.add_argument('--clahe', action='store_true',
                       help='Apply CLAHE for local contrast enhancement')
    parser.add_argument('--dehaze-strength', type=float, default=1.0,
                       help='Dehazing strength (0.5-2.0, default: 1.0)')
    parser.add_argument('--clahe-clip', type=float, default=2.0,
                       help='CLAHE clip limit (1.0-4.0, default: 2.0)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("Underwater Fish Dataset Preprocessing")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Dehazing: {'Yes' if args.dehaze else 'No'}", end='')
    if args.dehaze:
        print(f" (strength: {args.dehaze_strength})")
    else:
        print()
    print(f"CLAHE: {'Yes' if args.clahe else 'No'}", end='')
    if args.clahe:
        print(f" (clip limit: {args.clahe_clip})")
    else:
        print()
    print("=" * 60)
    
    if not args.dehaze and not args.clahe:
        print("\nWarning: No preprocessing options selected.")
        print("Images will be copied without modification.")
        print("Use --dehaze and/or --clahe to apply enhancements.")
    
    # Run preprocessing
    preprocess_dataset(
        args.input,
        args.output,
        apply_dehaze=args.dehaze,
        apply_clahe_flag=args.clahe,
        dehaze_strength=args.dehaze_strength,
        clahe_clip=args.clahe_clip
    )


if __name__ == '__main__':
    main()