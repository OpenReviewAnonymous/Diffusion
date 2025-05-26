#!/bin/bash

# Define the seeds to use
SEEDS=(1020 1021 1022 1023 1024 1025 1026 1027 1028 1029)

# Original file and temporary file
ORIGINAL_FILE="univariate/adaptive_diffwave.py"
TEMP_FILE="univariate/adaptive_diffwave_temp.py"

# Backup the original file
cp "$ORIGINAL_FILE" "$ORIGINAL_FILE.backup"

# Record the start time
echo "Start running multi-seed adaptive DiffWave experiments: $(date)"
echo "The following seeds will be used: ${SEEDS[@]}"

# Train with each seed
for seed in "${SEEDS[@]}"; do
    echo "============================================="
    echo "Start training with seed $seed"
    echo "============================================="
    
    # Modify the seed value in the Python file
    sed -e "s/np.random.seed(1010)/np.random.seed($seed)/" \
        -e "s/torch.manual_seed(1010)/torch.manual_seed($seed)/" \
        "$ORIGINAL_FILE" > "$TEMP_FILE"
    
    # Use the modified file
    mv "$TEMP_FILE" "$ORIGINAL_FILE"
    
    # Run the Python script
    python "$ORIGINAL_FILE"
    
    echo "Training with seed $seed is complete"
    echo "============================================="
done

# Restore the original file
mv "$ORIGINAL_FILE.backup" "$ORIGINAL_FILE"

echo "All seed trainings are complete: $(date)"
