#!/usr/bin/env python3
"""
Compare YOLO checkpoint files to identify structural differences
Shows what's stored in each checkpoint and why sizes differ
"""

import torch
import sys
from pathlib import Path


def analyze_checkpoint(checkpoint_path):
    """Analyze a YOLO checkpoint and return detailed structure info"""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'path': str(checkpoint_path),
            'file_size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
            'top_level_keys': list(ckpt.keys()) if isinstance(ckpt, dict) else ['Not a dict'],
            'details': {}
        }
        
        if isinstance(ckpt, dict):
            for key, value in ckpt.items():
                detail = {
                    'type': type(value).__name__,
                }
                
                # Analyze based on type
                if isinstance(value, dict):
                    detail['num_items'] = len(value)
                    detail['keys_sample'] = list(value.keys())[:5]  # First 5 keys
                    
                    # Check if it's a state_dict with tensors
                    if value and isinstance(next(iter(value.values())), torch.Tensor):
                        tensors = list(value.values())
                        detail['tensor_info'] = {
                            'num_tensors': len(tensors),
                            'first_tensor_dtype': str(tensors[0].dtype),
                            'first_tensor_shape': list(tensors[0].shape),
                            'dtypes': list(set(str(t.dtype) for t in tensors if isinstance(t, torch.Tensor))),
                        }
                        
                        # Calculate size
                        total_size = sum(t.element_size() * t.nelement() for t in tensors if isinstance(t, torch.Tensor))
                        detail['tensor_size_mb'] = total_size / (1024 * 1024)
                
                elif isinstance(value, torch.nn.Module):
                    detail['module_class'] = value.__class__.__name__
                    try:
                        state_dict = value.state_dict()
                        detail['num_params'] = len(state_dict)
                        tensors = list(state_dict.values())
                        if tensors:
                            detail['param_dtype'] = str(tensors[0].dtype)
                            total_size = sum(t.element_size() * t.nelement() for t in tensors)
                            detail['param_size_mb'] = total_size / (1024 * 1024)
                    except:
                        detail['state_dict'] = 'Could not extract'
                
                elif isinstance(value, torch.Tensor):
                    detail['shape'] = list(value.shape)
                    detail['dtype'] = str(value.dtype)
                    detail['size_mb'] = (value.element_size() * value.nelement()) / (1024 * 1024)
                
                elif value is not None:
                    detail['value_preview'] = str(value)[:100]
                
                info['details'][key] = detail
        
        return info
    
    except Exception as e:
        return {
            'path': str(checkpoint_path),
            'error': str(e)
        }


def compare_checkpoints(good_checkpoint, bad_checkpoint):
    """Compare two checkpoints and show differences"""
    
    print("=" * 100)
    print("CHECKPOINT COMPARISON")
    print("=" * 100)
    print()
    
    good_info = analyze_checkpoint(good_checkpoint)
    bad_info = analyze_checkpoint(bad_checkpoint)
    
    # Show basic info
    print(f"GOOD CHECKPOINT (reference):")
    print(f"  Path: {good_info['path']}")
    print(f"  Size: {good_info['file_size_mb']:.2f} MB")
    print()
    
    print(f"BAD CHECKPOINT (our custom one):")
    print(f"  Path: {bad_info['path']}")
    print(f"  Size: {bad_info['file_size_mb']:.2f} MB")
    print(f"  Size difference: {bad_info['file_size_mb'] - good_info['file_size_mb']:.2f} MB")
    print()
    
    print("=" * 100)
    print("TOP-LEVEL KEYS COMPARISON")
    print("=" * 100)
    print()
    
    good_keys = set(good_info.get('top_level_keys', []))
    bad_keys = set(bad_info.get('top_level_keys', []))
    
    print(f"Keys in GOOD: {sorted(good_keys)}")
    print(f"Keys in BAD:  {sorted(bad_keys)}")
    print()
    
    if good_keys == bad_keys:
        print("✓ Same top-level keys")
    else:
        only_good = good_keys - bad_keys
        only_bad = bad_keys - good_keys
        if only_good:
            print(f"⚠ Only in GOOD: {sorted(only_good)}")
        if only_bad:
            print(f"⚠ Only in BAD:  {sorted(only_bad)}")
    
    print()
    print("=" * 100)
    print("DETAILED KEY ANALYSIS")
    print("=" * 100)
    print()
    
    # Compare each key
    common_keys = good_keys & bad_keys
    
    for key in sorted(common_keys):
        good_detail = good_info['details'].get(key, {})
        bad_detail = bad_info['details'].get(key, {})
        
        print(f"KEY: '{key}'")
        print(f"  GOOD: {good_detail.get('type', 'N/A')}")
        print(f"  BAD:  {bad_detail.get('type', 'N/A')}")
        
        # Type mismatch
        if good_detail.get('type') != bad_detail.get('type'):
            print(f"  ⚠ TYPE MISMATCH!")
            print(f"     GOOD is {good_detail.get('type')}")
            print(f"     BAD is {bad_detail.get('type')}")
        
        # For dicts with tensors
        if 'tensor_info' in good_detail and 'tensor_info' in bad_detail:
            good_ti = good_detail['tensor_info']
            bad_ti = bad_detail['tensor_info']
            
            print(f"  Tensors:")
            print(f"    GOOD: {good_ti['num_tensors']} tensors, {good_ti['tensor_size_mb']:.2f} MB, dtype: {good_ti['dtypes']}")
            print(f"    BAD:  {bad_ti['num_tensors']} tensors, {bad_ti['tensor_size_mb']:.2f} MB, dtype: {bad_ti['dtypes']}")
            
            if good_ti['tensor_size_mb'] != bad_ti['tensor_size_mb']:
                diff = bad_ti['tensor_size_mb'] - good_ti['tensor_size_mb']
                print(f"    ⚠ SIZE DIFFERENCE: {diff:+.2f} MB")
                
                # Check dtype difference
                if good_ti['dtypes'] != bad_ti['dtypes']:
                    print(f"    ⚠ DTYPE MISMATCH!")
                    print(f"       GOOD uses: {good_ti['dtypes']}")
                    print(f"       BAD uses:  {bad_ti['dtypes']}")
                    print(f"       This is likely the cause of size difference!")
        
        # For modules
        elif 'param_size_mb' in good_detail and 'param_size_mb' in bad_detail:
            print(f"  Module params:")
            print(f"    GOOD: {good_detail['num_params']} params, {good_detail['param_size_mb']:.2f} MB, dtype: {good_detail.get('param_dtype')}")
            print(f"    BAD:  {bad_detail['num_params']} params, {bad_detail['param_size_mb']:.2f} MB, dtype: {bad_detail.get('param_dtype')}")
        
        print()
    
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    
    # Find the smoking gun
    print("Most likely causes of size difference:")
    
    for key in sorted(common_keys):
        good_detail = good_info['details'].get(key, {})
        bad_detail = bad_info['details'].get(key, {})
        
        # Check tensor dtypes
        if 'tensor_info' in good_detail and 'tensor_info' in bad_detail:
            good_dtypes = set(good_detail['tensor_info']['dtypes'])
            bad_dtypes = set(bad_detail['tensor_info']['dtypes'])
            
            if good_dtypes != bad_dtypes:
                print(f"  ⚠ '{key}': dtype mismatch")
                print(f"     GOOD: {good_dtypes}")
                print(f"     BAD:  {bad_dtypes}")
                
                good_size = good_detail['tensor_info']['tensor_size_mb']
                bad_size = bad_detail['tensor_info']['tensor_size_mb']
                print(f"     Size impact: {bad_size - good_size:+.2f} MB")
                print()
    
    print("=" * 100)
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python compare_checkpoints.py <good_checkpoint.pt> <bad_checkpoint.pt>")
        print()
        print("Example:")
        print("  python compare_checkpoints.py best.pt best_prec.pt")
        sys.exit(1)
    
    good_path = Path(sys.argv[1])
    bad_path = Path(sys.argv[2])
    
    if not good_path.exists():
        print(f"Error: {good_path} not found")
        sys.exit(1)
    
    if not bad_path.exists():
        print(f"Error: {bad_path} not found")
        sys.exit(1)
    
    compare_checkpoints(good_path, bad_path)
