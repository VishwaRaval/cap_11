#!/usr/bin/env python3
"""
Check dataset quality after preprocessing.
Identifies images that might be too poor quality even after dehazing.

Usage:
    python check_image_quality.py --dataset dataset_root_dehazed_aggressive
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def calculate_image_quality_metrics(image_path):
    """
    Calculate quality metrics for underwater images.
    Returns metrics that indicate if image is too poor quality.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert to different color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    metrics = {}
    
    # 1. Contrast (Laplacian variance - higher = more edges/detail)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    metrics['contrast'] = laplacian_var
    
    # 2. Brightness (average L channel in LAB)
    metrics['brightness'] = np.mean(lab[:,:,0])
    
    # 3. Color saturation (average S channel in HSV)
    metrics['saturation'] = np.mean(hsv[:,:,1])
    
    # 4. Blue tint dominance (check if still too blue after preprocessing)
    b, g, r = cv2.split(img)
    blue_dominance = np.mean(b) - np.mean(r)
    metrics['blue_tint'] = blue_dominance
    
    # 5. Histogram spread (check if image uses full dynamic range)
    hist_spread = np.std(gray)
    metrics['dynamic_range'] = hist_spread
    
    return metrics


def flag_poor_quality_images(dataset_root, split='train'):
    """
    Flag images that are still poor quality after preprocessing.
    """
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / split / 'images'
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    print(f"\n{'='*70}")
    print(f"ANALYZING {split.upper()} SET IMAGE QUALITY")
    print(f"{'='*70}")
    print(f"Total images: {len(image_files)}\n")
    
    poor_quality = []
    quality_stats = {
        'contrast': [],
        'brightness': [],
        'saturation': [],
        'blue_tint': [],
        'dynamic_range': []
    }
    
    for img_path in tqdm(image_files, desc=f"Analyzing {split}"):
        metrics = calculate_image_quality_metrics(img_path)
        if metrics is None:
            continue
        
        # Collect stats
        for key in quality_stats:
            quality_stats[key].append(metrics[key])
        
        # Flag poor quality images based on thresholds
        flags = []
        
        # Very low contrast (blurry/hazy)
        if metrics['contrast'] < 50:
            flags.append('low_contrast')
        
        # Too dark or too bright
        if metrics['brightness'] < 80 or metrics['brightness'] > 200:
            flags.append('poor_brightness')
        
        # Very low saturation (washed out)
        if metrics['saturation'] < 30:
            flags.append('washed_out')
        
        # Still heavily blue tinted after preprocessing
        if metrics['blue_tint'] > 30:
            flags.append('excessive_blue_tint')
        
        # Poor dynamic range (flat image)
        if metrics['dynamic_range'] < 30:
            flags.append('flat_histogram')
        
        if flags:
            poor_quality.append({
                'path': str(img_path),
                'name': img_path.name,
                'flags': flags,
                'metrics': metrics
            })
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"QUALITY STATISTICS")
    print(f"{'='*70}")
    
    for metric_name, values in quality_stats.items():
        values = np.array(values)
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean:   {np.mean(values):.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  Std:    {np.std(values):.2f}")
        print(f"  Min:    {np.min(values):.2f}")
        print(f"  Max:    {np.max(values):.2f}")
    
    # Print poor quality summary
    print(f"\n{'='*70}")
    print(f"POOR QUALITY IMAGES")
    print(f"{'='*70}")
    print(f"Total flagged: {len(poor_quality)} ({len(poor_quality)/len(image_files)*100:.1f}%)\n")
    
    if poor_quality:
        # Count by flag type
        flag_counts = {}
        for img in poor_quality:
            for flag in img['flags']:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        print("Issues breakdown:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  {flag:25s}: {count:4d} images")
        
        # Save list to file
        output_file = dataset_root / f'{split}_poor_quality.txt'
        with open(output_file, 'w') as f:
            f.write("# Poor Quality Images After Preprocessing\n")
            f.write("# Format: filename | flags | contrast | brightness | saturation | blue_tint\n\n")
            
            for img in sorted(poor_quality, key=lambda x: len(x['flags']), reverse=True):
                flags_str = ', '.join(img['flags'])
                m = img['metrics']
                line = f"{img['name']} | {flags_str} | {m['contrast']:.1f} | {m['brightness']:.1f} | {m['saturation']:.1f} | {m['blue_tint']:.1f}\n"
                f.write(line)
        
        print(f"\n✓ Saved poor quality list to: {output_file}")
        print(f"\nWorst 10 images:")
        for i, img in enumerate(sorted(poor_quality, key=lambda x: len(x['flags']), reverse=True)[:10], 1):
            print(f"  {i:2d}. {img['name']:50s} | {', '.join(img['flags'])}")
    else:
        print("✓ No poor quality images found!")
    
    print(f"{'='*70}\n")
    
    return poor_quality, quality_stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze image quality after preprocessing"
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to preprocessed dataset root')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'valid', 'test'],
                       help='Dataset splits to analyze (default: train valid test)')
    
    args = parser.parse_args()
    
    for split in args.splits:
        poor_quality, stats = flag_poor_quality_images(args.dataset, split)
    
    print("\n💡 RECOMMENDATIONS:")
    print("="*70)
    print("If many images are flagged:")
    print("  1. Your preprocessing might need to be MORE aggressive")
    print("  2. Some images might be unsalvageable (consider removing)")
    print("  3. Dataset might benefit from manual curation")
    print("\nIf few images are flagged:")
    print("  ✓ Preprocessing worked well!")
    print("  ✓ Proceed with training on this dataset")
    print("="*70)


if __name__ == '__main__':
    main()
