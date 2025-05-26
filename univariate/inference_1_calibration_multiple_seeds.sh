#!/bin/bash

# Create directories
mkdir -p results logs

# Set seed range
START_SEED=1020
END_SEED=1029

# Record start time
echo "Start multi-seed calibration and inference - $(date)"
echo "Using seed range: $START_SEED to $END_SEED"

# Create header for result CSV file
echo "seed,mu,phi,omega,alpha,beta" > results/calibration_results.csv

# Process each seed
for SEED in $(seq $START_SEED $END_SEED); do
    echo "===== Processing seed: $SEED ====="
    
    # 1. Run calibration process
    echo "1. Running calibration process..."
    python -m univariate.calibrate_garch --seed $SEED --save
    
    # 2. Get calibrated parameters
    if [ -f "results/params_seed_${SEED}.json" ]; then
        # Extract parameter values from JSON file
        MU=$(cat results/params_seed_${SEED}.json | grep -oP '"mu": \K[^,}]*')
        PHI=$(cat results/params_seed_${SEED}.json | grep -oP '"phi": \K[^,}]*')
        OMEGA=$(cat results/params_seed_${SEED}.json | grep -oP '"omega": \K[^,}]*')
        ALPHA=$(cat results/params_seed_${SEED}.json | grep -oP '"alpha": \K[^,}]*')
        BETA=$(cat results/params_seed_${SEED}.json | grep -oP '"beta": \K[^,}]*')
        
        # Add to CSV file
        echo "$SEED,$MU,$PHI,$OMEGA,$ALPHA,$BETA" >> results/calibration_results.csv
        
        # 3. Generate inference samples
        echo "3. Generating inference samples..."
        python -m univariate.evaluate_garch --mu $MU --phi $PHI --omega $OMEGA --alpha $ALPHA --beta $BETA --seed $SEED
    else
        echo "Error: Parameter file does not exist results/params_seed_${SEED}.json"
    fi
    
    echo "Finished processing seed $SEED"
    echo ""
done

# Create summary plots
echo "Generating parameter summary plots..."
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read results
df = pd.read_csv('results/calibration_results.csv')

# True parameter values
true_values = {
    'mu': 0.0,
    'phi': 0.95,
    'omega': 0.05,
    'alpha': 0.05,
    'beta': 0.90
}

# Calculate means
print('Parameter means:')
for param in ['mu', 'phi', 'omega', 'alpha', 'beta']:
    print(f'Mean {param}: {df[param].mean()}, True value: {true_values[param]}')

# Plot parameter distributions
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

params_list = ['mu', 'phi', 'omega', 'alpha', 'beta']
for i, param in enumerate(params_list):
    if i < len(axes):
        ax = axes[i]
        ax.bar(df['seed'], df[param])
        ax.axhline(y=true_values[param], color='r', linestyle='-', label='True Value')
        ax.set_title(f'{param.capitalize()} Estimates')
        ax.set_xlabel('Seed')
        ax.set_ylabel('Value')
        ax.legend()

plt.tight_layout()
plt.savefig('results/parameter_estimates.png')
print('Parameter estimate plot saved to results/parameter_estimates.png')
"

echo "All processing completed - $(date)"
