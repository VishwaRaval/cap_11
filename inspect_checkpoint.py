#!/usr/bin/env python3
"""
Inspect YOLO checkpoint file structure
This can be run on your HPC system with torch
"""

import torch
import sys
from pathlib import Path


def inspect_checkpoint(path):
    """Show detailed structure of a checkpoint"""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    
    print(f"=" * 80)
    print(f"Checkpoint: {path.name}")
    print(f"Size: {path.stat().st_size / (1024*1024):.2f} MB")
    print(f"=" * 80)
    print()
    
    if not isinstance(ckpt, dict):
        print(f"Not a dict! Type: {type(ckpt)}")
        return
    
    print(f"Top-level keys: {list(ckpt.keys())}")
    print()
    
    for key, value in ckpt.items():
        print(f"KEY: '{key}'")
        print(f"  Type: {type(value).__name__}")
        
        if value is None:
            print(f"  Value: None")
        
        elif isinstance(value, dict):
            print(f"  Dict with {len(value)} items")
            
            # Check if it's a state_dict
            if value:
                first_val = next(iter(value.values()))
                if isinstance(first_val, torch.Tensor):
                    tensors = [v for v in value.values() if isinstance(v, torch.Tensor)]
                    
                    # Get all dtypes
                    dtypes = set(str(t.dtype) for t in tensors)
                    
                    # Calculate total size
                    total_elements = sum(t.nelement() for t in tensors)
                    total_bytes = sum(t.element_size() * t.nelement() for t in tensors)
                    
                    print(f"  Contains {len(tensors)} tensors:")
                    print(f"    Total elements: {total_elements:,}")
                    print(f"    Total size: {total_bytes / (1024*1024):.2f} MB")
                    print(f"    Data types: {dtypes}")
                    
                    # Show example tensor
                    first_tensor = tensors[0]
                    print(f"    Example tensor:")
                    print(f"      Shape: {list(first_tensor.shape)}")
                    print(f"      Dtype: {first_tensor.dtype}")
                    print(f"      Size: {first_tensor.element_size() * first_tensor.nelement() / 1024:.2f} KB")
        
        elif isinstance(value, torch.nn.Module):
            print(f"  Module: {value.__class__.__name__}")
            state = value.state_dict()
            tensors = list(state.values())
            total_bytes = sum(t.element_size() * t.nelement() for t in tensors if isinstance(t, torch.Tensor))
            print(f"    Parameters: {len(tensors)}")
            print(f"    Size: {total_bytes / (1024*1024):.2f} MB")
        
        elif isinstance(value, (int, float, str, bool)):
            print(f"  Value: {value}")
        
        else:
            print(f"  Value: {str(value)[:100]}")
        
        print()
    
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint.pt>")
        print()
        print("Example:")
        print("  python inspect_checkpoint.py best.pt")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
    
    inspect_checkpoint(path)
