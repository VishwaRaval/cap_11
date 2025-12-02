#!/usr/bin/env python3
"""
LCH-Based Underwater Preprocessing
Uses LCH color space to directly manipulate blue hue for better color correction

LCH is superior for underwater correction because:
- Hue (H) allows direct targeting of blue tones (210-270°)
- More intuitive color manipulation than LAB
- Preserves image structure while correcting color cast

Usage:
    python preprocess_underwater_lch.py \
        --input dataset_root \
        --output dataset_root_lch_corrected \
        --blue-shift-strength 0.7
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


def bgr_to_lch(image):
    """
    Convert BGR to LCH color space.
    OpenCV doesn't have direct BGR->LCH, so we go BGR->LAB->LCH
    """
    # BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    
    # LAB to LCH
    # C (Chroma) = sqrt(A^2 + B^2)
    # H (Hue) = atan2(B, A) in radians, convert to degrees
    C = np.sqrt(A**2 + B**2)
    H = np.arctan2(B, A) * 180 / np.pi  # Convert to degrees
    
    # Normalize H to 0-360 range
    H = np.where(H < 0, H + 360, H)
    
    return L, C, H


def lch_to_bgr(L, C, H):
    """
    Convert LCH back to BGR color space.
    """
    # LCH to LAB
    # A = C * cos(H)
    # B = C * sin(H)
    H_rad = H * np.pi / 180  # Convert to radians
    A = C * np.cos(H_rad)
    B = C * np.sin(H_rad)
    
    # Merge LAB channels
    lab = cv2.merge([L, A, B]).astype(np.uint8)
    
    # LAB to BGR
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return bgr


def correct_underwater_blue_hue(L, C, H, blue_shift_strength=0.7, target_neutral_boost=0.3):
    """
    Correct blue hue in underwater images using LCH color space.
    
    Args:
        L: Lightness channel
        C: Chroma channel
        H: Hue channel (0-360 degrees)
        blue_shift_strength: How much to shift blue hues (0-1)
        target_neutral_boost: How much to boost warm colors (0-1)
    
    Blue hues in underwater images typically fall in:
    - Cyan-Blue: 180-240°
    - Blue: 240-270°
    - Blue-Magenta: 270-300°
    
    Strategy:
    1. Identify blue/cyan pixels
    2. Shift their hue toward green/yellow (more neutral)
    3. Slightly reduce their chroma (desaturation)
    4. Boost warm colors to compensate
    """
    H_corrected = H.copy()
    C_corrected = C.copy()
    
    # Define blue hue ranges (in degrees)
    cyan_blue_start = 180
    blue_end = 300
    
    # Create mask for blue/cyan pixels
    blue_mask = (H >= cyan_blue_start) & (H <= blue_end)
    
    # Calculate how "blue" each pixel is (0 at edges, 1 at pure blue 240°)
    blue_center = 240
    blue_intensity = np.zeros_like(H)
    blue_intensity[blue_mask] = 1.0 - np.abs(H[blue_mask] - blue_center) / (blue_end - cyan_blue_start)
    blue_intensity = np.clip(blue_intensity, 0, 1)
    
    # Shift blue hues toward green/yellow (reduce hue angle)
    # Blue (240°) -> Cyan-Green (160-180°)
    hue_shift_amount = -60 * blue_shift_strength * blue_intensity
    H_corrected = H + hue_shift_amount
    
    # Wrap hue to 0-360 range
    H_corrected = np.where(H_corrected < 0, H_corrected + 360, H_corrected)
    H_corrected = np.where(H_corrected > 360, H_corrected - 360, H_corrected)
    
    # Reduce chroma (saturation) of blue pixels to make them less dominant
    chroma_reduction = 0.7 * blue_intensity
    C_corrected[blue_mask] = C[blue_mask] * (1 - chroma_reduction[blue_mask])
    
    # Boost warm colors (red-yellow: 0-60°, 300-360°) to compensate
    warm_mask = ((H >= 0) & (H <= 60)) | ((H >= 300) & (H <= 360))
    C_corrected[warm_mask] = np.clip(C[warm_mask] * (1 + target_neutral_boost), 0, 255)
    
    return L, C_corrected, H_corrected


def enhance_contrast_lch(L, C, H, clip_limit=4.0, tile_size=8):
    """
    Apply CLAHE to Lightness channel in LCH space.
    """
    # Convert L to uint8 for CLAHE
    L_uint8 = L.astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    L_enhanced = clahe.apply(L_uint8).astype(np.float32)
    
    return L_enhanced, C, H


def process_underwater_lch(image, blue_shift_strength=0.7, clahe_clip=4.0):
    """
    Complete underwater correction pipeline using LCH color space.
    """
    # Stage 1: Convert to LCH
    L, C, H = bgr_to_lch(image)
    
    # Stage 2: Correct blue hue
    L, C_corrected, H_corrected = correct_underwater_blue_hue(
        L, C, H, 
        blue_shift_strength=blue_shift_strength,
        target_neutral_boost=0.3
    )
    
    # Stage 3: Enhance contrast (CLAHE on L channel)
    L_enhanced, C_final, H_final = enhance_contrast_lch(
        L, C_corrected, H_corrected,
        clip_limit=clahe_clip,
        tile_size=8
    )
    
    # Stage 4: Convert back to BGR
    result = lch_to_bgr(L_enhanced, C_final, H_final)
    
    # Stage 5: Slight sharpening
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]]) * 0.15
    sharpened = cv2.filter2D(result, -1, kernel)
    
    # Blend
    final = cv2.addWeighted(result, 0.75, sharpened, 0.25, 0)
    
    return final


def process_image(input_path, output_path, blue_shift_strength=0.7, clahe_clip=4.0):
    """
    Process a single image with LCH-based correction.
    """
    # Read image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Warning: Could not read {input_path}")
        return False
    
    # Apply LCH correction
    enhanced = process_underwater_lch(img, blue_shift_strength, clahe_clip)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), enhanced)
    return True


def preprocess_dataset(input_root, output_root, blue_shift_strength=0.7, clahe_clip=4.0):
    """
    Preprocess entire dataset with LCH-based correction.
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
            
            success = process_image(img_path, output_img_path, blue_shift_strength, clahe_clip)
            if success:
                total_processed += 1
            else:
                total_failed += 1
            
            # Copy label
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                output_label_path = output_labels_dir / label_path.name
                shutil.copy2(label_path, output_label_path)
    
    print(f"\n{'='*70}")
    print(f"LCH PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {total_processed} images")
    if total_failed > 0:
        print(f"✗ Failed: {total_failed} images")
    print(f"✓ Output saved to: {output_root}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LCH-based underwater preprocessing for blue tint correction"
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input dataset root directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for preprocessed dataset')
    parser.add_argument('--blue-shift-strength', type=float, default=0.7,
                       help='Blue hue shift strength (0-1, default: 0.7)')
    parser.add_argument('--clahe-clip', type=float, default=4.0,
                       help='CLAHE clip limit (default: 4.0)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LCH-BASED UNDERWATER PREPROCESSING")
    print("="*70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"\nSettings:")
    print(f"  Blue shift strength: {args.blue_shift_strength}")
    print(f"  CLAHE clip limit:    {args.clahe_clip}")
    print("\nEnhancements applied:")
    print("  1. BGR -> LCH color space conversion")
    print("  2. Blue hue shift (240° -> 180°, Blue -> Cyan-Green)")
    print("  3. Blue chroma reduction (desaturation)")
    print("  4. Warm color boost (red-yellow enhancement)")
    print("  5. CLAHE on Lightness channel")
    print("  6. Sharpening for edge enhancement")
    print("  7. LCH -> BGR conversion")
    print("="*70 + "\n")
    
    # Run preprocessing
    preprocess_dataset(
        args.input, 
        args.output,
        blue_shift_strength=args.blue_shift_strength,
        clahe_clip=args.clahe_clip
    )
    
    print("\n💡 NEXT STEPS:")
    print("-"*70)
    print("1. Check quality:")
    print(f"   python check_image_quality.py --dataset {args.output}")
    print("")
    print("2. Compare blue tint metric:")
    print("   Target: < 50 (currently 122)")
    print("")
    print("3. If blue tint < 50: Train immediately!")
    print("4. If still high: Increase --blue-shift-strength to 0.9")
    print("-"*70)


if __name__ == '__main__':
    main()
