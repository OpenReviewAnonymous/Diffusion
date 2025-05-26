#!/bin/bash

# Script to run experiments for different values of lambda_1 and lambda_2
# The adaptive_ddpm.py script reads HP_LAMBDA_1, HP_LAMBDA_2, and RUN_ID environment variables.

PYTHON_SCRIPT="adaptive_ddpm.py"

echo "Running experiments for effects of regularity terms..."

# Experiment Group 1: Vary lambda_1, lambda_2 = 0
echo "--- Experiment Group 1: Vary lambda_1, lambda_2 = 0 ---"
LAMBDA1_VALUES=("5e-3"  "7.5e-3")
LAMBDA2_VALUE="0"

for lambda1 in "${LAMBDA1_VALUES[@]}"; do
    RUN_ID="lambda1_${lambda1}_lambda2_${LAMBDA2_VALUE}"
    echo "Running with HP_LAMBDA_1=${lambda1}, HP_LAMBDA_2=${LAMBDA2_VALUE}, RUN_ID=${RUN_ID}"
    export HP_LAMBDA_1="${lambda1}"
    export HP_LAMBDA_2="${LAMBDA2_VALUE}"
    export RUN_ID="${RUN_ID}"
    CUDA_VISIBLE_DEVICES=0 /the_path_to_python/python "${PYTHON_SCRIPT}"
    echo "Finished run for lambda1=${lambda1}, lambda2=${lambda2}"
    echo ""
done

