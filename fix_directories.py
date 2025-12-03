#!/usr/bin/env python3
"""
Fix directory paths in experiment scripts to use absolute paths
"""

import re
from pathlib import Path

# Target directory
TARGET_DIR = "/scratch/am14419/projects/cap_11/runs/detect"

# Scripts to fix
scripts = [
    'run_experiment_1.sh',
    'run_experiment_2.sh', 
    'run_experiment_3.sh',
    'run_all_experiments.sh',
    'train_yolo11_fish_enhanced_fixed.py',
]

def fix_script(script_path):
    """Fix directory paths in a script."""
    script_path = Path(script_path)
    
    if not script_path.exists():
        print(f"⚠️  Skipping {script_path.name} (not found)")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix OUTPUT_DIR lines in bash scripts
    if script_path.suffix == '.sh':
        content = re.sub(
            r'OUTPUT_DIR="runs/detect/\$\{EXPERIMENT_NAME\}"',
            f'OUTPUT_DIR="{TARGET_DIR}/${{EXPERIMENT_NAME}}"',
            content
        )
        
        # Fix project path in training commands
        content = re.sub(
            r'--project \$\{OUTPUT_DIR\}/runs',
            '--project ${OUTPUT_DIR}',
            content
        )
        
    # Fix Python scripts
    elif script_path.suffix == '.py':
        # Fix project default
        content = re.sub(
            r"'project', type=str, default='runs/detect'",
            f"'project', type=str, default='{TARGET_DIR}'",
            content
        )
        
        # Fix any hardcoded paths
        content = re.sub(
            r"Path\('runs/detect'\)",
            f"Path('{TARGET_DIR}')",
            content
        )
    
    # Only write if changed
    if content != original_content:
        with open(script_path, 'w') as f:
            f.write(content)
        print(f"✅ Fixed {script_path.name}")
        return True
    else:
        print(f"✓  {script_path.name} already correct")
        return False

def main():
    print("=" * 80)
    print("FIXING DIRECTORY PATHS IN EXPERIMENT SCRIPTS")
    print("=" * 80)
    print(f"\nTarget directory: {TARGET_DIR}\n")
    
    fixed = 0
    for script in scripts:
        if fix_script(script):
            fixed += 1
    
    print("\n" + "=" * 80)
    print(f"✓ Fixed {fixed} script(s)")
    print("=" * 80)
    print(f"\nAll experiment outputs will now go to:")
    print(f"  {TARGET_DIR}")
    print()

if __name__ == '__main__':
    main()
