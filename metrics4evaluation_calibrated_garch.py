#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics calculation and visualization for Calibrated GARCH model (only comparing real data and Calibrated GARCH)
"""

# =============== Import necessary libraries and functions ===============
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, entropy
from sklearn.neighbors import KernelDensity
import ot  # Required: pip install pot
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import os

# Import common functions from metrics4evaluation.py

from metrics4evaluation import (
    load_datasets, # Need to load original data to get scaler and N_test
    plot_correlation_matrix,
    plot_covariance_matrix,
    plot_statistical_distributions,
    data_summary,
    merge_images,
    compare_datasets, # Use this function to compute metrics
    _to_numpy,
    compute_average_signature,
    path_signature_distance,
    # Other metric functions are also called by compare_datasets, but are imported directly for clarity or future direct use
    correlation_matrix_difference,
    marginal_wasserstein_distance,
    sliced_wasserstein_2d,
    conditional_wasserstein_distance,
    # KL divergence
    kl_divergence
)



timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")

# =============== Main function (only evaluate Calibrated GARCH) ===============
def main():
    """Main function: load data, compute evaluation metrics and visualize (only for real data and Calibrated GARCH)"""
    print("Start loading datasets...")
    # Use the original load_datasets function to load all data, then select only real and calibrated data
    real_data, diffwave_data, adaptive_diffwave_data, diffwave_model_name, adaptive_model_name, calibrated_data, ddpm_data, ddpm_model_name = load_datasets()

    # Create directory to save figures
    save_dir = "evaluation_figures_calibrated_garch"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory for saving figures: {save_dir}")

    # Define display names and file names for models
    calibrated_display_name = 'Calibrated AR-GARCH'
    calibrated_file_name = 'calibrated_model'
    real_display_name = 'Real Data'
    real_file_name = 'Real_Data'


    # Compute unified data ranges (considering only real and calibrated data)
    print("\nComputing unified visualization ranges...")

    # Compute unified range for correlation matrices
    corr_matrices = [
        np.corrcoef(np.hstack([real_data[:, :, 0], real_data[:, :, 1]]), rowvar=False),
        np.corrcoef(np.hstack([calibrated_data[:, :, 0], calibrated_data[:, :, 1]]), rowvar=False),
    ]
    corr_min = min(np.min(matrix) for matrix in corr_matrices)
    corr_max = max(np.max(matrix) for matrix in corr_matrices)
    print(f"Unified range for correlation matrices: [{corr_min:.4f}, {corr_max:.4f}]")

    # Compute unified range for covariance matrices
    cov_matrices = [
        np.cov(np.hstack([real_data[:, :, 0], real_data[:, :, 1]]), rowvar=False),
        np.cov(np.hstack([calibrated_data[:, :, 0], calibrated_data[:, :, 1]]), rowvar=False),
    ]
    cov_min = min(np.min(matrix) for matrix in cov_matrices)
    cov_max = max(np.max(matrix) for matrix in cov_matrices)
    print(f"Unified range for covariance matrices: [{cov_min:.4f}, {cov_max:.4f}]")

    # Compute unified range for statistical distribution plots
    # Mean
    means_x_all = [
        np.mean(real_data[:, :, 0], axis=1),
        np.mean(calibrated_data[:, :, 0], axis=1),
    ]
    means_y_all = [
        np.mean(real_data[:, :, 1], axis=1),
        np.mean(calibrated_data[:, :, 1], axis=1),
    ]
    mean_x_lim = (min(np.min(x) for x in means_x_all), max(np.max(x) for x in means_x_all))
    mean_y_lim = (min(np.min(y) for y in means_y_all), max(np.max(y) for y in means_y_all))

    # Variance
    vars_x_all = [
        np.var(real_data[:, :, 0], axis=1),
        np.var(calibrated_data[:, :, 0], axis=1),
    ]
    vars_y_all = [
        np.var(real_data[:, :, 1], axis=1),
        np.var(calibrated_data[:, :, 1], axis=1),
    ]
    var_x_lim = (min(np.min(x) for x in vars_x_all), max(np.max(x) for x in vars_x_all))
    var_y_lim = (min(np.min(y) for y in vars_y_all), max(np.max(y) for y in vars_y_all))

    # Skewness
    skew_x_all = [
        skew(real_data[:, :, 0], axis=1),
        skew(calibrated_data[:, :, 0], axis=1),
    ]
    skew_y_all = [
        skew(real_data[:, :, 1], axis=1),
        skew(calibrated_data[:, :, 1], axis=1),
    ]
    skew_x_lim = (min(np.min(x) for x in skew_x_all), max(np.max(x) for x in skew_x_all))
    skew_y_lim = (min(np.min(y) for y in skew_y_all), max(np.max(y) for y in skew_y_all))

    # Unified x-axis range for statistical distribution plots
    stat_xlims = [mean_x_lim, mean_y_lim, var_x_lim, var_y_lim, skew_x_lim, skew_y_lim]
    print(f"Unified range for statistical distribution plots computed")

    # Compute unified range for data summary plots
    x_all = np.concatenate([
        real_data[:, :, 0].flatten(),
        calibrated_data[:, :, 0].flatten(),
    ])
    y_all = np.concatenate([
        real_data[:, :, 1].flatten(),
        calibrated_data[:, :, 1].flatten(),
    ])

    x_lim = (np.min(x_all), np.max(x_all))
    y_lim = (np.min(y_all), np.max(y_all))

    summary_xlims = [x_lim, y_lim, x_lim, y_lim]

    # Basic data statistics, using unified range
    # Call data_summary and get the returned path
    real_summary_path = data_summary(real_data, real_display_name, save_dir=save_dir, xlims=summary_xlims)
    calibrated_summary_path = data_summary(calibrated_data, calibrated_display_name, save_dir=save_dir, xlims=summary_xlims)

    # Merge summary plots, using returned paths
    summary_paths = [
        real_summary_path,
        calibrated_summary_path
    ]
    merge_images(summary_paths, f"{save_dir}/merged_data_summaries_{calibrated_file_name}.png")
    print(f"Summary plots of real data and {calibrated_display_name} have been merged and saved to: {save_dir}/merged_data_summaries_{calibrated_file_name}.png")


    # Visualize correlation matrices, using unified range
    print("\nPlotting correlation matrices...")
    real_corr_path = f"{save_dir}/{real_file_name}_corr_matrix.png"
    calibrated_corr_path = f"{save_dir}/{calibrated_file_name}_corr_matrix.png"

    plot_correlation_matrix(real_data, f"{real_display_name} Correlation Matrix", real_corr_path, vmin=corr_min, vmax=corr_max)
    plot_correlation_matrix(calibrated_data, f"{calibrated_display_name} Correlation Matrix", calibrated_corr_path, vmin=corr_min, vmax=corr_max)

    # Merge correlation matrix plots
    corr_paths = [real_corr_path, calibrated_corr_path]
    merge_images(corr_paths, f"{save_dir}/merged_correlation_matrices_{calibrated_file_name}.png")

    # Visualize covariance matrices, using unified range
    print("\nPlotting covariance matrices...")
    real_cov_path = f"{save_dir}/{real_file_name}_cov_matrix.png"
    calibrated_cov_path = f"{save_dir}/{calibrated_file_name}_cov_matrix.png"

    plot_covariance_matrix(real_data, f"{real_display_name} Covariance Matrix", real_cov_path, vmin=cov_min, vmax=cov_max)
    plot_covariance_matrix(calibrated_data, f"{calibrated_display_name} Covariance Matrix", calibrated_cov_path, vmin=cov_min, vmax=cov_max)

    # Merge covariance matrix plots
    cov_paths = [real_cov_path, calibrated_cov_path]
    merge_images(cov_paths, f"{save_dir}/merged_covariance_matrices_{calibrated_file_name}.png")

    # Visualize statistical distributions, using unified range
    print("\nPlotting statistical distributions...")
    real_stats_path = f"{save_dir}/{real_file_name}_stats_dist.png"
    calibrated_stats_path = f"{save_dir}/{calibrated_file_name}_stats_dist.png"

    plot_statistical_distributions(real_data, f"{real_display_name} Statistical Distributions", real_stats_path, xlims=stat_xlims)
    plot_statistical_distributions(calibrated_data, f"{calibrated_display_name} Statistical Distributions", calibrated_stats_path, xlims=stat_xlims)

    # Merge statistical distribution plots
    stats_paths = [real_stats_path, calibrated_stats_path]
    merge_images(stats_paths, f"{save_dir}/merged_statistical_distributions_{calibrated_file_name}.png")


    # Compute evaluation metrics (only compare real data and Calibrated GARCH)
    print(f"\nComputing evaluation metrics ({calibrated_display_name} vs {real_display_name})...")
    calibrated_metrics = compare_datasets(real_data, calibrated_data, f"{calibrated_display_name} model")

    # Print evaluation results
    print(f"\n====== {calibrated_display_name} Evaluation Results ======")
    for metric, value in calibrated_metrics.items():
         if not np.isnan(value):
            print(f"{metric}: {value:.6f}")
         else:
            print(f"{metric}: Calculation failed or not performed")

    # Save results as a text file
    with open(f"{save_dir}/evaluation_results_{calibrated_file_name}.txt", "w") as f:
        f.write(f"====== {calibrated_display_name} vs {real_display_name} Performance Evaluation Results ======\n\n")
        f.write(f"{calibrated_display_name} model evaluation results:\n")
        for metric, value in calibrated_metrics.items():
             if not np.isnan(value):
                f.write(f"{metric}: {value:.6f}\n")
             else:
                f.write(f"{metric}: Calculation failed or not performed\n")


    print(f"Evaluation results have been saved to: {save_dir}/evaluation_results_{calibrated_file_name}.txt")

if __name__ == "__main__":
    main()
