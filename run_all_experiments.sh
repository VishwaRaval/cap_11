#!/bin/bash
# Master Script: Run All 3 Experiments Sequentially
# Total time: ~12-18 hours (4-6 hours per experiment)

set -e  # Exit on error

echo "========================================================================"
echo "AUTOMATED EXPERIMENT SUITE - UNDERWATER FISH DETECTION"
echo "========================================================================"
echo ""
echo "Running 3 experiments to reach 70% accuracy target:"
echo ""
echo "  Experiment 1: LAB + Conservative (batch 32)"
echo "               → Expected: 68-71%"
echo ""
echo "  Experiment 2: LAB + Ultra-Conservative (batch 48)"
echo "               → Expected: 69-72%"
echo ""
echo "  Experiment 3: LAB Aggressive + Conservative (batch 32)"
echo "               → Expected: 68-73%"
echo ""
echo "Total estimated time: 12-18 hours on A100-80GB"
echo "========================================================================"
echo ""

# Configuration
export ORIGINAL_DATASET="dataset_root"
export BASE_MODEL="runs/detect/best.pt_s_cosine_finetune_v1/weights/best.pt"

# Verify paths exist
if [ ! -d "$ORIGINAL_DATASET" ]; then
    echo "❌ ERROR: Original dataset not found at: $ORIGINAL_DATASET"
    echo "   Please edit run_all_experiments.sh and set correct path"
    exit 1
fi

if [ ! -f "$BASE_MODEL" ]; then
    echo "❌ ERROR: Base model not found at: $BASE_MODEL"
    echo "   Please edit run_all_experiments.sh and set correct path"
    exit 1
fi

# Create experiments directory
mkdir -p runs/detect

# Log file
MASTER_LOG="runs/detect/master_log.txt"
echo "Master experiment log - Started at $(date)" > ${MASTER_LOG}

# Function to run experiment with error handling
run_experiment() {
    local exp_num=$1
    local exp_script=$2
    local exp_name=$3
    
    echo "" | tee -a ${MASTER_LOG}
    echo "========================================================================" | tee -a ${MASTER_LOG}
    echo "STARTING EXPERIMENT ${exp_num}: ${exp_name}" | tee -a ${MASTER_LOG}
    echo "Started at: $(date)" | tee -a ${MASTER_LOG}
    echo "========================================================================" | tee -a ${MASTER_LOG}
    echo "" | tee -a ${MASTER_LOG}
    
    # Run experiment
    if bash ${exp_script}; then
        echo "" | tee -a ${MASTER_LOG}
        echo "✓ Experiment ${exp_num} completed successfully at $(date)" | tee -a ${MASTER_LOG}
        echo "" | tee -a ${MASTER_LOG}
    else
        echo "" | tee -a ${MASTER_LOG}
        echo "❌ Experiment ${exp_num} failed at $(date)" | tee -a ${MASTER_LOG}
        echo "   Check logs in experiments/exp${exp_num}_*/training.log" | tee -a ${MASTER_LOG}
        echo "" | tee -a ${MASTER_LOG}
        
        # Ask user if they want to continue
        read -p "Continue with remaining experiments? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Experiments stopped by user" | tee -a ${MASTER_LOG}
            exit 1
        fi
    fi
}

# Run all experiments
echo "Starting automated experiment suite..."
echo ""

run_experiment 1 "run_experiment_1.sh" "LAB + Conservative (Batch 32)"
run_experiment 2 "run_experiment_2.sh" "LAB + Ultra-Conservative (Batch 48)"
run_experiment 3 "run_experiment_3.sh" "LAB Aggressive + Conservative"

# Final summary
echo ""
echo "========================================================================"
echo "🎉 ALL EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo ""
echo "Completed at: $(date)" | tee -a ${MASTER_LOG}
echo ""
echo "Results summary:"
echo "  Experiment 1: runs/detect/exp1_lab_conservative_b32/weights/best.pt"
echo "  Experiment 2: runs/detect/exp2_lab_ultraconservative_b48/weights/best.pt"
echo "  Experiment 3: runs/detect/exp3_lab_aggressive_conservative/weights/best.pt"
echo ""
echo "Next steps:"
echo "  1. Check training logs in each experiment directory"
echo "  2. Compare metrics_summary.csv in each runs/training folder"
echo "  3. Review which experiment achieved highest accuracy"
echo "  4. Select best model for deployment"
echo ""
echo "Quick accuracy check:"
echo "  grep 'avg_accuracy' runs/detect/exp*/training.log | tail -3"
echo ""
echo "========================================================================"
