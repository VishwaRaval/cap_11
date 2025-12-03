#!/usr/bin/env python3
"""
Test different LAB correction parameters on sample images
Helps find optimal blue_reduction value before processing entire dataset

Usage:
    python test_lab_parameters.py \
        --input /path/to/dataset/train/images \
        --samples 5 \
        --output test_lab_results
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random


def calculate_blue_tint(image):
    """Calculate blue tint metric."""
    if len(image.shape) == 3:
        blue_channel = image[:, :, 0]
        return np.mean(blue_channel)
    return 0.0


def lab_underwater_correction(image, blue_reduction=0.7, gamma=1.0):
    """LAB color correction - same as main script."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    if gamma != 1.0:
        l = l.astype(np.float32) / 255.0
        l = np.power(l, gamma)
        l = (l * 255.0).astype(np.uint8)
    
    b_mean = np.mean(b)
    b_shift = (128 - b_mean) * (1 - blue_reduction)
    b_corrected = b.astype(np.float32) + b_shift
    b_corrected = np.clip(b_corrected, 0, 255).astype(np.uint8)
    
    lab_corrected = cv2.merge([l, a, b_corrected])
    corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
    
    return corrected


def test_parameters(image_path, output_dir, blue_reductions, gammas):
    """Test different parameter combinations on a single image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to read: {image_path}")
        return
    
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blue_original = calculate_blue_tint(image)
    
    # Test different blue_reduction values
    n_blue = len(blue_reductions)
    n_gamma = len(gammas)
    
    fig, axes = plt.subplots(n_gamma, n_blue + 1, figsize=(4*(n_blue+1), 4*n_gamma))
    
    if n_gamma == 1:
        axes = axes.reshape(1, -1)
    
    for i, gamma in enumerate(gammas):
        # Show original in first column
        axes[i, 0].imshow(original_rgb)
        axes[i, 0].set_title(f'Original\nBlue: {blue_original:.1f}')
        axes[i, 0].axis('off')
        
        # Show corrected versions
        for j, blue_red in enumerate(blue_reductions):
            corrected = lab_underwater_correction(image, blue_red, gamma)
            corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
            blue_after = calculate_blue_tint(corrected)
            
            axes[i, j+1].imshow(corrected_rgb)
            title = f'BR={blue_red:.2f}, γ={gamma:.1f}\nBlue: {blue_after:.1f}'
            title += f'\n({((blue_original-blue_after)/blue_original*100):.0f}% ↓)'
            axes[i, j+1].set_title(title, fontsize=9)
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f"{Path(image_path).stem}_parameter_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test LAB correction parameters on sample images"
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input images directory')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of random samples to test (default: 5)')
    parser.add_argument('--output', type=str, default='test_lab_results',
                       help='Output directory for test visualizations')
    parser.add_argument('--blue-reductions', type=str, default='0.5,0.6,0.7,0.8',
                       help='Comma-separated blue_reduction values to test (default: 0.5,0.6,0.7,0.8)')
    parser.add_argument('--gammas', type=str, default='1.0',
                       help='Comma-separated gamma values to test (default: 1.0)')
    
    args = parser.parse_args()
    
    # Parse parameters
    blue_reductions = [float(x) for x in args.blue_reductions.split(',')]
    gammas = [float(x) for x in args.gammas.split(',')]
    
    print("\n" + "="*70)
    print("LAB PARAMETER TESTING")
    print("="*70)
    print(f"📂 Input: {args.input}")
    print(f"📂 Output: {args.output}")
    print(f"🎨 Testing blue_reduction values: {blue_reductions}")
    print(f"💡 Testing gamma values: {gammas}")
    print(f"🖼️  Samples: {args.samples}")
    print("="*70 + "\n")
    
    # Get image files
    input_dir = Path(args.input)
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    # Sample random images
    sample_images = random.sample(image_files, min(args.samples, len(image_files)))
    
    print(f"Testing on {len(sample_images)} random images...\n")
    
    for img_path in sample_images:
        print(f"Processing: {img_path.name}")
        test_parameters(img_path, args.output, blue_reductions, gammas)
    
    print("\n✅ Parameter testing complete!")
    print(f"📁 Results saved to: {args.output}")
    print("\n📋 Next steps:")
    print("1. Review the visualizations")
    print("2. Choose the blue_reduction value that gives best color correction")
    print("   - Look for fish that are clearly visible against background")
    print("   - Avoid over-correction (fish should still look natural)")
    print("3. Use chosen value with preprocess_lab_underwater.py")
    print("\n💡 Recommended starting point: blue_reduction=0.7")


if __name__ == '__main__':
    main()
