#!/usr/bin/env python3
"""
Test script to verify multi-checkpoint saving works correctly

This runs a short 5-epoch test training to verify:
1. All 4 checkpoints are created
2. Callback fires and saves best_prec.pt and best_rec.pt
3. YOLO saves best.pt and last.pt as expected

Usage:
    python test_checkpoint_saving.py
"""

import sys
from pathlib import Path
import shutil

def test_checkpoint_saving():
    """Run a minimal training test to verify checkpoint saving"""
    
    print("=" * 80)
    print("CHECKPOINT SAVING TEST")
    print("=" * 80)
    print("\nThis test will:")
    print("  1. Run 5 epochs of training")
    print("  2. Verify all 4 checkpoints are created")
    print("  3. Show which epochs each checkpoint was saved")
    print("\n" + "=" * 80 + "\n")
    
    # Check if dataset exists
    dataset_root = Path("/scratch/am14419/projects/cap_11/dataset_root")
    if not dataset_root.exists():
        print(f"❌ Dataset not found: {dataset_root}")
        print("   Please update the path in this script")
        return False
    
    # Set up test parameters
    test_config = {
        'data': str(dataset_root),
        'model': 'yolo11n.pt',  # Use nano for fastest test
        'epochs': 5,
        'batch': 16,
        'patience': 50,
        'workers': 4,
        'hyp': 'hyp_precision_focus_v1.yaml',
        'name': 'checkpoint_test'
    }
    
    print("Test configuration:")
    for key, value in test_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Import and run training
    try:
        from train_ultra_stable_v2 import train_ultra_stable
        import argparse
        
        # Create args object
        args = argparse.Namespace(**test_config)
        
        print("🚀 Starting test training...\n")
        results_dir = train_ultra_stable(args)
        
        # Verify checkpoints
        print("\n" + "=" * 80)
        print("CHECKPOINT VERIFICATION")
        print("=" * 80 + "\n")
        
        weights_dir = results_dir / 'weights'
        expected_checkpoints = ['best.pt', 'best_prec.pt', 'best_rec.pt', 'last.pt']
        
        all_exist = True
        for ckpt in expected_checkpoints:
            ckpt_path = weights_dir / ckpt
            if ckpt_path.exists():
                size_mb = ckpt_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {ckpt:15s} ({size_mb:.1f} MB)")
            else:
                print(f"  ❌ {ckpt:15s} (MISSING!)")
                all_exist = False
        
        # Check tracker file
        tracker_file = weights_dir / 'checkpoint_tracker.txt'
        if tracker_file.exists():
            print(f"\n📊 Checkpoint Tracker:")
            with open(tracker_file, 'r') as f:
                print("  " + "\n  ".join(f.read().strip().split('\n')))
        else:
            print(f"\n⚠️  checkpoint_tracker.txt not found")
        
        print("\n" + "=" * 80)
        if all_exist:
            print("✅ TEST PASSED: All 4 checkpoints saved successfully!")
            print("=" * 80)
            print("\nThe multi-checkpoint system is working correctly.")
            print("You can now safely run full training experiments.")
            
            # Ask if user wants to delete test
            print(f"\nTest results saved to: {results_dir}")
            response = input("\nDelete test directory? (yes/no): ")
            if response.lower() == 'yes':
                shutil.rmtree(results_dir)
                print("✓ Test directory deleted")
            
            return True
        else:
            print("❌ TEST FAILED: Some checkpoints are missing!")
            print("=" * 80)
            print("\nTroubleshooting:")
            print("  1. Check if callback is registered properly")
            print("  2. Verify metrics are being calculated")
            print("  3. Look for error messages during training")
            print(f"  4. Check logs in: {results_dir}")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_checkpoint_saving()
    sys.exit(0 if success else 1)
