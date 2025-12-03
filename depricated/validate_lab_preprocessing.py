#!/usr/bin/env python3
"""
Validate LAB preprocessing quality by comparing original vs corrected datasets

Usage:
    python validate_lab_preprocessing.py \
        --original /path/to/original_dataset \
        --corrected /path/to/lab_corrected_dataset
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_metrics(image):
    """Calculate image quality metrics."""
    metrics = {}
    
    # Blue tint
    if len(image.shape) == 3:
        blue_channel = image[:, :, 0]
        metrics['blue_tint'] = np.mean(blue_channel)
        
        # Color balance
        b, g, r = cv2.split(image)
        metrics['mean_blue'] = np.mean(b)
        metrics['mean_green'] = np.mean(g)
        metrics['mean_red'] = np.mean(r)
        
        # Color cast (B-R difference)
        metrics['blue_cast'] = metrics['mean_blue'] - metrics['mean_red']
    
    # Brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    metrics['brightness'] = np.mean(gray)
    
    # Contrast (std of grayscale)
    metrics['contrast'] = np.std(gray)
    
    return metrics


def validate_datasets(original_root, corrected_root):
    """Compare original and corrected datasets."""
    original_root = Path(original_root)
    corrected_root = Path(corrected_root)
    
    results = {
        'original': [],
        'corrected': [],
        'improvements': 0,
        'worse': 0,
        'total': 0,
    }
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        orig_images = original_root / split / 'images'
        corr_images = corrected_root / split / 'images'
        
        if not orig_images.exists() or not corr_images.exists():
            continue
        
        # Get matching images
        orig_files = {f.name: f for f in orig_images.glob('*.jpg')}
        orig_files.update({f.name: f for f in orig_images.glob('*.png')})
        
        print(f"\nValidating {split} split: {len(orig_files)} images")
        
        for img_name, orig_path in tqdm(orig_files.items(), desc=split):
            corr_path = corr_images / img_name
            
            if not corr_path.exists():
                print(f"⚠️  Missing corrected image: {img_name}")
                continue
            
            # Read images
            orig_img = cv2.imread(str(orig_path))
            corr_img = cv2.imread(str(corr_path))
            
            if orig_img is None or corr_img is None:
                continue
            
            # Calculate metrics
            orig_metrics = calculate_metrics(orig_img)
            corr_metrics = calculate_metrics(corr_img)
            
            results['original'].append(orig_metrics)
            results['corrected'].append(corr_metrics)
            results['total'] += 1
            
            # Check if improved (lower blue tint is better)
            if corr_metrics['blue_tint'] < orig_metrics['blue_tint']:
                results['improvements'] += 1
            else:
                results['worse'] += 1
    
    return results


def print_validation_report(results):
    """Print comprehensive validation report."""
    print("\n" + "="*70)
    print("LAB PREPROCESSING VALIDATION REPORT")
    print("="*70)
    
    if results['total'] == 0:
        print("❌ No images to validate!")
        return
    
    # Calculate averages
    orig_metrics = {}
    corr_metrics = {}
    
    for key in results['original'][0].keys():
        orig_values = [m[key] for m in results['original']]
        corr_values = [m[key] for m in results['corrected']]
        
        orig_metrics[key] = {
            'mean': np.mean(orig_values),
            'std': np.std(orig_values),
            'min': np.min(orig_values),
            'max': np.max(orig_values),
        }
        
        corr_metrics[key] = {
            'mean': np.mean(corr_values),
            'std': np.std(corr_values),
            'min': np.min(corr_values),
            'max': np.max(corr_values),
        }
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total images validated: {results['total']}")
    print(f"  Images improved:        {results['improvements']} ({results['improvements']/results['total']*100:.1f}%)")
    print(f"  Images worse:           {results['worse']} ({results['worse']/results['total']*100:.1f}%)")
    
    print(f"\n🎨 Blue Tint Analysis:")
    print(f"  ORIGINAL:")
    print(f"    Mean:  {orig_metrics['blue_tint']['mean']:.1f}")
    print(f"    Range: {orig_metrics['blue_tint']['min']:.1f} - {orig_metrics['blue_tint']['max']:.1f}")
    print(f"  CORRECTED:")
    print(f"    Mean:  {corr_metrics['blue_tint']['mean']:.1f}")
    print(f"    Range: {corr_metrics['blue_tint']['min']:.1f} - {corr_metrics['blue_tint']['max']:.1f}")
    
    blue_reduction = ((orig_metrics['blue_tint']['mean'] - corr_metrics['blue_tint']['mean']) 
                     / orig_metrics['blue_tint']['mean'] * 100)
    print(f"  ✅ Blue tint reduced by: {blue_reduction:.1f}%")
    
    print(f"\n🌈 Color Balance Analysis:")
    print(f"  ORIGINAL:")
    print(f"    Blue:  {orig_metrics['mean_blue']['mean']:.1f}")
    print(f"    Green: {orig_metrics['mean_green']['mean']:.1f}")
    print(f"    Red:   {orig_metrics['mean_red']['mean']:.1f}")
    print(f"    Blue cast (B-R): {orig_metrics['blue_cast']['mean']:.1f}")
    print(f"  CORRECTED:")
    print(f"    Blue:  {corr_metrics['mean_blue']['mean']:.1f}")
    print(f"    Green: {corr_metrics['mean_green']['mean']:.1f}")
    print(f"    Red:   {corr_metrics['mean_red']['mean']:.1f}")
    print(f"    Blue cast (B-R): {corr_metrics['blue_cast']['mean']:.1f}")
    
    print(f"\n💡 Brightness & Contrast:")
    print(f"  ORIGINAL:")
    print(f"    Brightness: {orig_metrics['brightness']['mean']:.1f} ± {orig_metrics['brightness']['std']:.1f}")
    print(f"    Contrast:   {orig_metrics['contrast']['mean']:.1f} ± {orig_metrics['contrast']['std']:.1f}")
    print(f"  CORRECTED:")
    print(f"    Brightness: {corr_metrics['brightness']['mean']:.1f} ± {corr_metrics['brightness']['std']:.1f}")
    print(f"    Contrast:   {corr_metrics['contrast']['mean']:.1f} ± {corr_metrics['contrast']['std']:.1f}")
    
    # Quality assessment
    print(f"\n✅ Quality Assessment:")
    blue_tint_threshold = 50
    good_images = sum(1 for m in results['corrected'] if m['blue_tint'] <= blue_tint_threshold)
    pct_good = (good_images / results['total']) * 100
    print(f"  Images with blue tint < {blue_tint_threshold}: {good_images}/{results['total']} ({pct_good:.1f}%)")
    
    if pct_good >= 90:
        print(f"  ✅ EXCELLENT - {pct_good:.0f}% of images meet quality threshold")
    elif pct_good >= 70:
        print(f"  ⚠️  GOOD - {pct_good:.0f}% of images meet quality threshold")
    else:
        print(f"  ❌ POOR - Only {pct_good:.0f}% of images meet quality threshold")
        print(f"     Consider increasing blue_reduction parameter")
    
    print("="*70 + "\n")
    
    return orig_metrics, corr_metrics


def plot_distribution_comparison(results, output_path):
    """Plot distribution comparison of key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Blue tint distribution
    orig_blue = [m['blue_tint'] for m in results['original']]
    corr_blue = [m['blue_tint'] for m in results['corrected']]
    
    axes[0, 0].hist(orig_blue, bins=50, alpha=0.6, label='Original', color='blue')
    axes[0, 0].hist(corr_blue, bins=50, alpha=0.6, label='Corrected', color='orange')
    axes[0, 0].axvline(x=50, color='red', linestyle='--', label='Target threshold')
    axes[0, 0].set_xlabel('Blue Tint Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Blue Tint Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Blue cast distribution
    orig_cast = [m['blue_cast'] for m in results['original']]
    corr_cast = [m['blue_cast'] for m in results['corrected']]
    
    axes[0, 1].hist(orig_cast, bins=50, alpha=0.6, label='Original', color='blue')
    axes[0, 1].hist(corr_cast, bins=50, alpha=0.6, label='Corrected', color='orange')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Neutral')
    axes[0, 1].set_xlabel('Blue Cast (B-R)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Color Balance Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Brightness distribution
    orig_bright = [m['brightness'] for m in results['original']]
    corr_bright = [m['brightness'] for m in results['corrected']]
    
    axes[1, 0].hist(orig_bright, bins=50, alpha=0.6, label='Original', color='blue')
    axes[1, 0].hist(corr_bright, bins=50, alpha=0.6, label='Corrected', color='orange')
    axes[1, 0].set_xlabel('Brightness')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Brightness Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Contrast distribution
    orig_contrast = [m['contrast'] for m in results['original']]
    corr_contrast = [m['contrast'] for m in results['corrected']]
    
    axes[1, 1].hist(orig_contrast, bins=50, alpha=0.6, label='Original', color='blue')
    axes[1, 1].hist(corr_contrast, bins=50, alpha=0.6, label='Corrected', color='orange')
    axes[1, 1].set_xlabel('Contrast (Std Dev)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Contrast Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Distribution plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate LAB preprocessing quality"
    )
    
    parser.add_argument('--original', type=str, required=True,
                       help='Path to original dataset')
    parser.add_argument('--corrected', type=str, required=True,
                       help='Path to LAB corrected dataset')
    parser.add_argument('--output', type=str, default='validation_report.png',
                       help='Output path for distribution plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VALIDATING LAB PREPROCESSING")
    print("="*70)
    print(f"📂 Original:  {args.original}")
    print(f"📂 Corrected: {args.corrected}")
    print("="*70 + "\n")
    
    # Validate
    results = validate_datasets(args.original, args.corrected)
    
    # Print report
    orig_metrics, corr_metrics = print_validation_report(results)
    
    # Plot distributions
    plot_distribution_comparison(results, args.output)
    
    print("✅ Validation complete!")


if __name__ == '__main__':
    main()
