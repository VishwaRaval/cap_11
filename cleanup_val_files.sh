#!/bin/bash
# Clean up unnecessary validation prediction files created by YOLO training
# These files are just visualization outputs and not needed for deployment

echo "════════════════════════════════════════════════════════════"
echo "YOLO Validation Files Cleanup Script"
echo "════════════════════════════════════════════════════════════"
echo ""

# Base directory for all YOLO runs
RUNS_DIR="/scratch/am14419/projects/cap_11/runs/detect"

if [ ! -d "$RUNS_DIR" ]; then
    echo "❌ Runs directory not found: $RUNS_DIR"
    exit 1
fi

echo "Searching for validation prediction files in: $RUNS_DIR"
echo ""

# Find all val* files (validation predictions) - images only
VAL_FILES=$(find "$RUNS_DIR" -type f \( -name "val*.jpg" -o -name "val*.png" \) 2>/dev/null)
VAL_COUNT=$(echo "$VAL_FILES" | grep -v '^$' | wc -l)

if [ "$VAL_COUNT" -eq 0 ]; then
    echo "✅ No validation files found. Nothing to clean up!"
    exit 0
fi

# Calculate total size
TOTAL_SIZE=$(find "$RUNS_DIR" -type f \( -name "val*.jpg" -o -name "val*.png" \) -exec du -ch {} + 2>/dev/null | grep total$ | awk '{print $1}')

echo "Found $VAL_COUNT validation prediction files"
echo "Total size: $TOTAL_SIZE"
echo ""
echo "These files are:"
echo "  • Validation prediction visualizations (bounding boxes on images)"
echo "  • Created when save=True in YOLO training config"
echo "  • NOT needed for training, evaluation, or deployment"
echo "  • Safe to delete"
echo ""
echo "Will keep:"
echo "  • All .pt checkpoint files (best.pt, best_prec.pt, best_rec.pt, last.pt)"
echo "  • All .csv files (results.csv, metrics_summary.csv)"
echo "  • All plots (confusion_matrix.png, results.png, etc.)"
echo ""

# Ask for confirmation
read -p "Delete validation image files? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

echo ""
echo "🗑️  Deleting validation files..."

# Delete all val* prediction images (but not .pt, .csv, or plot files)
find "$RUNS_DIR" -type f \( -name "val*.jpg" -o -name "val*.png" \) -delete

echo ""
echo "✅ Cleanup complete!"
echo "   Deleted: $VAL_COUNT image files"
echo "   Freed: $TOTAL_SIZE"
echo ""
echo "✓ All checkpoint files (.pt) are safe and untouched"
echo "✓ All metrics files (.csv) are safe and untouched"
echo "✓ All plots (.png charts) are safe and untouched"
echo "════════════════════════════════════════════════════════════"
