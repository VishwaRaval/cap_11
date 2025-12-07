#!/usr/bin/env python3
"""
Setup Validation Script
Checks if all dependencies and files are correctly configured
"""

import sys
from pathlib import Path
import json

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLO',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install -r requirements.txt --break-system-packages")
        return False
    
    return True


def check_gpu():
    """Check CUDA/GPU availability"""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print(f"  ⚠ CUDA not available - using CPU (will be slower)")
            return False
    except:
        print(f"  ✗ Could not check GPU")
        return False


def check_files():
    """Check if required files exist"""
    print("\nChecking project files...")
    
    required_files = [
        'main.py',
        'ensemble_predictor.py',
        'dataset_labeler.py',
        'edge_deployer.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for filename in required_files:
        if Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_config():
    """Check if config file exists and is valid"""
    print("\nChecking configuration...")
    
    config_path = Path('config.json')
    if not config_path.exists():
        print(f"  ⚠ config.json not found")
        print(f"    Copy config.json.template to config.json and update paths")
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"  ✓ config.json found and valid")
        
        # Check required fields
        required_fields = ['dataset', 'models', 'output']
        for field in required_fields:
            if field in config:
                print(f"    ✓ {field}")
            else:
                print(f"    ✗ {field} - missing")
                return False
        
        return True
    except json.JSONDecodeError as e:
        print(f"  ✗ config.json is not valid JSON: {e}")
        return False


def check_models(config_path='config.json'):
    """Check if model files exist"""
    print("\nChecking model files...")
    
    if not Path(config_path).exists():
        print(f"  ⚠ No config file - skipping model check")
        return True  # Not critical
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        if 'models' not in config or 'top_5' not in config['models']:
            print(f"  ⚠ No models specified in config")
            return True
        
        models = config['models']['top_5']
        all_exist = True
        
        for model_path in models:
            if Path(model_path).exists():
                size_mb = Path(model_path).stat().st_size / (1024 * 1024)
                print(f"  ✓ {Path(model_path).name} ({size_mb:.2f} MB)")
            else:
                print(f"  ✗ {model_path} - NOT FOUND")
                all_exist = False
        
        return all_exist
    except:
        print(f"  ⚠ Could not check models")
        return True


def check_dataset(config_path='config.json'):
    """Check if dataset exists"""
    print("\nChecking dataset...")
    
    if not Path(config_path).exists():
        print(f"  ⚠ No config file - skipping dataset check")
        return True
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        if 'dataset' not in config:
            print(f"  ⚠ No dataset specified in config")
            return True
        
        dataset = config['dataset']
        
        # Check images directory
        if 'images' in dataset:
            img_dir = Path(dataset['images'])
            if img_dir.exists():
                num_images = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
                print(f"  ✓ Images directory ({num_images} images)")
            else:
                print(f"  ✗ Images directory not found: {img_dir}")
                return False
        
        # Check labels directory
        if 'labels' in dataset:
            lbl_dir = Path(dataset['labels'])
            if lbl_dir.exists():
                num_labels = len(list(lbl_dir.glob('*.txt')))
                print(f"  ✓ Labels directory ({num_labels} labels)")
            else:
                print(f"  ✗ Labels directory not found: {lbl_dir}")
                return False
        
        return True
    except:
        print(f"  ⚠ Could not check dataset")
        return True


def print_summary(checks):
    """Print summary of checks"""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name:30s} {status}")
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to go!")
        print("\nNext steps:")
        print("  1. Verify paths in config.json")
        print("  2. Run ensemble: python main.py ensemble --help")
        print("  3. Or run everything: python main.py all --help")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt --break-system-packages")
        print("  - Create config.json from config.json.template")
        print("  - Update paths in config.json to point to your files")
    
    return all_passed


def main():
    """Run all validation checks"""
    print("="*80)
    print("FISH DETECTION PROJECT - SETUP VALIDATION")
    print("="*80)
    
    checks = {
        'Dependencies': check_dependencies(),
        'GPU/CUDA': check_gpu(),
        'Project Files': check_files(),
        'Configuration': check_config(),
        'Model Files': check_models(),
        'Dataset': check_dataset()
    }
    
    all_passed = print_summary(checks)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
