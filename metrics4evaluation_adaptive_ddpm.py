# -*- coding: utf-8 -*-

"""
Adaptive DDPM model evaluation metrics calculation (only comparing real data and Adaptive DDPM) - optimized for hyperparameter search
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Removed plotting import
# import seaborn as sns # Removed plotting import
from scipy.stats import skew, entropy
from sklearn.neighbors import KernelDensity
import ot  # Required: pip install pot
from tqdm import tqdm
from datetime import datetime
# from PIL import Image # Removed plotting import
import os
import json # Import json library for structured output
import sys # Import sys module for stderr

# Assume metrics4evaluation contains the necessary functions and load_datasets
# If load_datasets is defined here, ensure it's correctly placed or imported
# For simplicity, assume it's imported or available in the scope
from metrics4evaluation import (
    load_datasets, # Ensure this function is available and works as expected
    # plot_correlation_matrix, # Plotting function disabled
    # plot_covariance_matrix, # Plotting function disabled
    # plot_statistical_distributions, # Plotting function disabled
    # data_summary, # Plotting function disabled
    # merge_images, # Plotting function disabled
    compare_datasets,
    _to_numpy,
    compute_average_signature,
    path_signature_distance,
    correlation_matrix_difference,
    marginal_wasserstein_distance,
    sliced_wasserstein_2d,
    conditional_wasserstein_distance,
    kl_divergence
)


# Function to calculate metrics
def calculate_metrics(generated_samples_path: str) -> dict:
    """Calculates evaluation metrics by comparing generated data to real data.

    Args:
        generated_samples_path: Path to the .npy file containing generated samples.

    Returns:
        A dictionary containing the calculated metrics, or a dict with an 'error' key if checks fail.
    """
    if not generated_samples_path or not os.path.exists(generated_samples_path):
        print(f"Error: Generated samples path is invalid or file does not exist: {generated_samples_path}", file=sys.stderr)
        return {"error": "Invalid generated samples path"}

    # Load real data - Removed try-except
    real_data = None
    datasets_tuple = load_datasets() # Or pass necessary args if required
    if datasets_tuple and len(datasets_tuple) > 0:
        real_data = datasets_tuple[0]
    else:
        print("Error: Could not load real data using load_datasets.", file=sys.stderr)
        return {"error": "Failed to load real data"}

    if real_data is None or (isinstance(real_data, np.ndarray) and real_data.size == 0):
         print("Error: Real data loaded is empty.", file=sys.stderr)
         return {"error": "Real data is empty"}

    # Ensure real_data is numpy array - Removed try-except
    if not isinstance(real_data, np.ndarray):
         real_data = np.array(real_data) # Direct conversion, will raise error if fails

    # Load generated Adaptive DDPM data - Removed try-except
    adaptive_diffwave_data = np.load(generated_samples_path)
    if adaptive_diffwave_data is None or adaptive_diffwave_data.size == 0:
         print(f"Error: Generated data from {generated_samples_path} is empty.", file=sys.stderr)
         return {"error": "Generated data is empty"}

    # Check dimensions (excluding batch size)
    if real_data.shape[1:] != adaptive_diffwave_data.shape[1:]:
         print(f"Warning: Real data shape {real_data.shape} and generated data shape {adaptive_diffwave_data.shape} mismatch beyond batch size.", file=sys.stderr)
         # Continue despite warning

    # Calculate metrics using compare_datasets - Removed try-except
    metrics_results = compare_datasets(real_data, adaptive_diffwave_data, "Adaptive DDPM")

    # Prepare JSON-compatible output
    printable_metrics = {}
    for k, v in metrics_results.items():
        if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
            if not np.isnan(v):
                 printable_metrics[k] = float(v)
        elif v is not None and not (isinstance(v, float) and np.isnan(v)):
             printable_metrics[k] = v

    return printable_metrics


# Main execution block (for standalone testing or use via CLI)
def main():
    """Main function: load data, calculate evaluation metrics and output results (only for real data and Adaptive DDPM)
       Expects generated samples path from environment variable or command line.
    """
    generated_samples_path = None
    if len(sys.argv) > 1:
        generated_samples_path = sys.argv[1]
        print(f"Using generated samples path from command line: {generated_samples_path}", file=sys.stderr)
    else:
        generated_samples_path = os.environ.get("GENERATED_SAMPLES_PATH")
        if generated_samples_path:
             print(f"Using generated samples path from environment variable: {generated_samples_path}", file=sys.stderr)

    if not generated_samples_path:
        print("Error: GENERATED_SAMPLES_PATH not provided via environment variable or command line argument.", file=sys.stderr)
        sys.exit(1)

    # Calculate metrics by calling the refactored function
    # Note: Errors inside calculate_metrics (without try-except) will now halt here.
    metrics = calculate_metrics(generated_samples_path)

    # Print results in JSON format to stdout
    if metrics and 'error' not in metrics:
        print("====== METRICS RESULTS ======", file=sys.stdout)
        print(json.dumps(metrics, indent=4, ensure_ascii=False), file=sys.stdout)
        print("====== END METRICS RESULTS ======", file=sys.stdout)
    else:
        print(f"Metrics calculation failed or returned error: {metrics.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1) # Exit with error code if calculation failed

if __name__ == "__main__":
    main()
