#!/bin/bash

# Script to run experiments for different values of lambda_1 and lambda_2
# The adaptive_ddpm.py script reads HP_LAMBDA_1, HP_LAMBDA_2, and RUN_ID environment variables.
# Modified to run experiments for different DDPM learning rates

PYTHON_SCRIPT="ddpm.py"

echo "Running experiments for different DDPM learning rates..."

# Experiment Group 1: Vary DDPM_LR
echo "--- Experiment Group 1: Vary DDPM_LR ---"
DDPM_LR_VALUES=("5e-4"  "1e-3") # Using a different variable name to avoid conflict


# Loop through DDPM_LR values
for lr in "${DDPM_LR_VALUES[@]}"; do
    RUN_ID="ddpm_lr_${lr}"
    echo "Running with HP_DDPM_LR=${lr}, RUN_ID=${RUN_ID}"
    export HP_DDPM_LR="${lr}"
    export RUN_ID="${RUN_ID}"
    CUDA_VISIBLE_DEVICES=4 /the_path_to_python/python "${PYTHON_SCRIPT}"
    echo "Finished run for DDPM_LR=${lr}"
    echo ""
done

