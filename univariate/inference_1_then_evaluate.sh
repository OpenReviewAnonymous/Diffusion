#!/bin/bash
# Univariate time series generation model evaluation script
# Evaluate multiple generated sample files and compare with real data

# Set environment variables if needed
# export PYTHONPATH=/path/to/add:$PYTHONPATH

# Use an array to store the filenames of generated samples to be evaluated
# These files should contain univariate time series data in numpy array format of shape (N,T) or (N,T,1)
FILES=(

    # "samples_adaptive_diffwave_2025-05-15_00-21.npy"

    
)

# Run the evaluation script, specify GPU device, and pass all files as arguments
echo "Starting univariate time series evaluation..."
CUDA_VISIBLE_DEVICES=7 python univariate/inference_and_metrics_1.py "${FILES[@]}"

# Find the latest log file
LATEST_LOG=$(ls -t logs/inference_and_metrics_*.log | head -1)

# Print the log file path for result viewing
echo "Evaluation completed. Latest log file path:"
echo $LATEST_LOG

# Extract and display the DataFrame section from the log file
echo "Results summary (DataFrame):"
sed -n '/EVALUATION RESULTS SUMMARY/,/END EVALUATION RESULTS SUMMARY/p' "$LATEST_LOG" | grep -v "EVALUATION RESULTS SUMMARY"

echo "Evaluation script finished!"




