#!/bin/bash
# Clean up unnecessary validation directories created by YOLO training
# These directories contain validation prediction visualizations and are not needed

echo "════════════════════════════════════════════════════════════"
echo "YOLO Validation Directories Cleanup Script"
echo "════════════════════════════════════════════════════════════"
echo ""

# Base directory for all YOLO runs
RUNS_DIR="/scratch/am14419/projects/cap_11/runs/detect"

if [ ! -d "$RUNS_DIR" ]; then
    echo "❌ Runs directory not found: $RUNS_DIR"
    exit 1
fi

echo "Searching for validation directories in: $RUNS_DIR"
echo ""

# Find all directories that start with "val"
VAL_DIRS=$(find "$RUNS_DIR" -type d -name "val*" 2>/dev/null)
VAL_COUNT=$(echo "$VAL_DIRS" | grep -v '^$' | wc -l)

if [ "$VAL_COUNT" -eq 0 ]; then
    echo "✅ No validation directories found. Nothing to clean up!"
    exit 0
fi

# Calculate total size
TOTAL_SIZE=$(du -ch $(find "$RUNS_DIR" -type d -name "val*" 2>/dev/null) 2>/dev/null | grep total$ | awk '{print $1}')

echo "Found $VAL_COUNT validation directories:"
echo "$VAL_DIRS" | while read dir; do
    if [ -n "$dir" ]; then
        dir_name=$(basename "$dir")
        parent=$(basename $(dirname "$dir"))
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  • $parent/$dir_name ($size)"
    fi
done

echo ""
echo "Total size: $TOTAL_SIZE"
echo ""
echo "These directories contain:"
echo "  • Validation prediction visualizations (bounding boxes on images)"
echo "  • Created during YOLO validation runs"
echo "  • NOT needed for training, evaluation, or deployment"
echo "  • Safe to delete"
echo ""
echo "Will keep:"
echo "  • All .pt checkpoint files (best.pt, best_prec.pt, best_rec.pt, last.pt)"
echo "  • All .csv files (results.csv, metrics_summary.csv)"
echo "  • All plots (confusion_matrix.png, results.png, etc.)"
echo "  • weights/ directories"
echo ""

# Ask for confirmation
read -p "Delete all validation directories? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

echo ""
echo "🗑️  Deleting validation directories..."

# Delete all directories starting with "val"
find "$RUNS_DIR" -type d -name "val*" -exec rm -rf {} + 2>/dev/null

echo ""
echo "✅ Cleanup complete!"
echo "   Deleted: $VAL_COUNT directories"
echo "   Freed: $TOTAL_SIZE"
echo ""
echo "✓ All checkpoint files (.pt) are safe and untouched"
echo "✓ All metrics files (.csv) are safe and untouched"
echo "✓ All plots (.png charts) are safe and untouched"
echo "════════════════════════════════════════════════════════════"
