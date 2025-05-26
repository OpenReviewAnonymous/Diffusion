#!/bin/bash

# Set Python path
PYTHON_PATH="python"
SCRIPT_PATH="univariate/adaptive_ddpm_regularity_test.py"

# Create log directory
LOG_DIR="logs_adaptive_ddpm_regularity_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "Starting Adaptive DDPM regularization comparison experiment"
echo "Logs will be saved to: $LOG_DIR"
echo "================================================"

# Function: run specified scheme
run_scheme() {
    local scheme=$1
    local gpu=$2
    local run_id=$3
    local seed=$4
    
    echo "Launching scheme $scheme run $run_id (GPU $gpu, SEED $seed)..."
    
    # Create log file name
    log_file="$LOG_DIR/scheme_${scheme}_run_${run_id}_gpu_${gpu}_seed_${seed}.log"
    
    # Use echo to automatically input selection, run the program, and pass the seed parameter
    CUDA_VISIBLE_DEVICES=$gpu echo "$scheme" | $PYTHON_PATH $SCRIPT_PATH --seed $seed > "$log_file" 2>&1 &
    
    echo "  PID: $! - Log: $log_file"
}

# Launch all tasks
echo ""
echo "Launching scheme A (GPU 0) - Off-diagonal Frobenius"
# Generate different random seeds for scheme A
seeds_A=(1001 1002 1003)
for i in {1..3}; do
    run_scheme "A" 0 $i ${seeds_A[$((i-1))]}
    sleep 2  # Avoid timestamp conflicts caused by simultaneous launches
done

echo ""
echo "Launching scheme B (GPU 6) - Off-diagonal Frobenius + log-det + τ tr"
# Generate different random seeds for scheme B
seeds_B=(1001 1002 1003)
for i in {1..3}; do
    run_scheme "B" 6 $i ${seeds_B[$((i-1))]}
    sleep 2
done

echo ""
echo "Launching scheme C (GPU 7) - Exponential distance-weighted L2"
# Generate different random seeds for scheme C
seeds_C=(1001 1002 1003)
for i in {1..3}; do
    run_scheme "C" 7 $i ${seeds_C[$((i-1))]}
    sleep 2
done

echo ""
echo "Launching scheme D (GPU 4) - Full Frobenius norm"
# Generate different random seeds for scheme D
seeds_D=(1001 1002 1003)
for i in {1..3}; do
    run_scheme "D" 4 $i ${seeds_D[$((i-1))]}
    sleep 2
done

echo ""
echo "================================================"
echo "All tasks have been launched!"
echo ""
echo "Monitoring commands:"
echo "  View all GPU usage: watch -n 1 nvidia-smi"
echo "  View running processes: ps aux | grep adaptive_ddpm"
echo "  Real-time view of a log: tail -f $LOG_DIR/scheme_A_run_1_gpu_1_seed_1001.log"
echo ""
echo "Waiting for all tasks to complete..."

# Wait for all background tasks to finish
wait

echo ""
echo "All tasks are completed!"
echo "Results are saved in: $LOG_DIR"

# Generate summary report
echo ""
echo "Generating summary report..."
summary_file="$LOG_DIR/summary_report.txt"

echo "Adaptive DDPM Regularization Scheme Comparison Experiment Summary" > $summary_file
echo "=================================" >> $summary_file
echo "Experiment time: $(date)" >> $summary_file
echo "" >> $summary_file

# Extract the final validation loss for each run
for scheme in A B C D; do
    echo "Scheme $scheme:" >> $summary_file
    for i in {1..3}; do
        # Select the corresponding seed array according to the scheme
        if [ "$scheme" == "A" ]; then
            seed=${seeds_A[$((i-1))]}
        elif [ "$scheme" == "B" ]; then
            seed=${seeds_B[$((i-1))]}
        elif [ "$scheme" == "C" ]; then
            seed=${seeds_C[$((i-1))]}
        else
            seed=${seeds_D[$((i-1))]}
        fi
        
        log_file="$LOG_DIR/scheme_${scheme}_run_${i}_gpu_*_seed_${seed}.log"
        if ls $log_file 1> /dev/null 2>&1; then
            # Extract the last validation loss
            final_loss=$(grep "【保存最佳模型】" $log_file | tail -1 | grep -oP '验证损失: \K[0-9.]+' || echo "N/A")
            echo "  Run $i (seed=$seed): Best validation loss = $final_loss" >> $summary_file
        else
            echo "  Run $i (seed=$seed): Log file not found" >> $summary_file
        fi
    done
    echo "" >> $summary_file
done

echo "Summary report has been saved to: $summary_file"
cat $summary_file
