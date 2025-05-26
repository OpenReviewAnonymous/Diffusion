# -*- coding: utf-8 -*-
import os
import re
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch # Might be needed indirectly by imported functions
import sys
import shutil # For cleaning up temp directory

# Add project root to path if necessary, or ensure modules are importable
# sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Example

# Import the refactored functions
from evaluate_adaptive_ddpm import run_evaluation
from metrics4evaluation_adaptive_ddpm import calculate_metrics


# Define the base directory for models if they are not in the current directory
# Set to None if the paths in the script are absolute or relative to CWD
MODEL_BASE_DIR = None # 

# Directory for temporary evaluation outputs
TEMP_OUTPUT_DIR = "temp_evaluation_outputs"

# --- Function to Parse Model Files ---
def get_model_files_from_script(script_path):
    """Parses .pth filenames listed anywhere in the specified Python script."""
    model_files = []
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
        potential_files = re.findall(r'^[ \t]*([\w\/\.-]*best_model_adaptive_ddpm[\w\/\.-]*\.pth)[ \t]*$', script_content, re.MULTILINE)
        model_files = [f.strip() for f in potential_files]

    hardcoded_models = [
        "best_model_adaptive_ddpm_20250511_191107.pth"
    ]
    combined_files = list(dict.fromkeys(model_files + hardcoded_models))
    print(f"Debug: Found {len(potential_files)} potential files via regex.")
    print(f"Debug: Using combined list of {len(combined_files)} unique models.")
    return combined_files

# --- Function to Extract Model Identifier ---
def extract_model_id(filename):
    """Extracts a sortable identifier from the model filename."""
    match_id = re.search(r'_(\d+)_2512376\.pth$', filename)
    if match_id:
        return int(match_id.group(1))

    match_ts = re.search(r'_(\d{8}_\d{6})|(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})', filename)
    if match_ts:
        return match_ts.group(1) or match_ts.group(2)

    base_name = os.path.basename(filename)
    fallback_id = base_name.replace("best_model_adaptive_ddpm_", "").replace(".pth", "")
    return fallback_id

# --- Main Logic ---
def main():
    """Main function to evaluate models, collect metrics, and plot results."""
    script_path = os.path.abspath(__file__)
    model_filenames = get_model_files_from_script(script_path)

    if not model_filenames:
        print("No model files (.pth) found from parsing or hardcoded list. Exiting.")
        return

    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    line_to_remove_index = -1
    for i, line in enumerate(lines):
        if '(model params)' in line and 'evaluate_adaptive_ddpm.py' in line and line.strip().startswith('Put'):
            line_to_remove_index = i
            break

    if line_to_remove_index != -1:
        print(f"Attempting to remove comment line {line_to_remove_index + 1} to avoid syntax errors...")
        del lines[line_to_remove_index]
        with open(script_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("Comment line likely removed. Please verify the file.")


    all_metrics_data = []
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

    print(f"Processing {len(model_filenames)} model files:")
    for i, fname in enumerate(model_filenames):
         print(f"- {fname}")
         if i > 15:
              print(f"  ... and {len(model_filenames) - i -1} more")
              break

    for model_filename in model_filenames:
        print(f"\n--- Processing model: {model_filename} ---")

        if MODEL_BASE_DIR:
            model_full_path = os.path.join(MODEL_BASE_DIR, model_filename)
        else:
            script_dir = os.path.dirname(script_path)
            path_relative_to_script = os.path.join(script_dir, model_filename)
            path_in_cwd = os.path.abspath(model_filename)

            if os.path.exists(path_relative_to_script):
                 model_full_path = path_relative_to_script
                 print(f"Found model relative to script: {model_full_path}")
            elif os.path.exists(path_in_cwd):
                 model_full_path = path_in_cwd
                 print(f"Found model in CWD: {model_full_path}")
            elif os.path.isabs(model_filename) and os.path.exists(model_filename):
                  model_full_path = model_filename
                  print(f"Using absolute path for model: {model_full_path}")
            else:
                  potential_diffusion_path = f"diffusion/{model_filename}"
                  if os.path.exists(potential_diffusion_path):
                       model_full_path = potential_diffusion_path
                       print(f"Assuming model in diffusion dir: {model_full_path}")
                  else:
                       print(f"Warning: Model file not found at likely locations: {path_relative_to_script}, {path_in_cwd}, or as absolute path {model_filename}. Skipping.")
                       all_metrics_data.append({'model_file': model_filename, 'error': 'File not found'})
                       continue

        generated_npy_path = None
        print(f"Running evaluation for {model_filename}...")
        generated_npy_path = run_evaluation(model_path=model_full_path, output_dir=TEMP_OUTPUT_DIR)
        print(f"Generated samples saved to: {generated_npy_path}")

        if generated_npy_path and os.path.exists(generated_npy_path):
            print(f"Calculating metrics for {generated_npy_path}...")
            metrics = calculate_metrics(generated_samples_path=generated_npy_path)

            if 'error' in metrics:
                 print(f"Metrics calculation failed: {metrics['error']}")
                 metrics_result = {'model_file': model_filename, 'error': metrics['error']}
            else:
                 print("Metrics calculated successfully.")
                 metrics_result = metrics
                 metrics_result['model_file'] = model_filename
            all_metrics_data.append(metrics_result)
        else:
             print("Error: Generated samples file not found after evaluation step.")
             all_metrics_data.append({'model_file': model_filename, 'error': 'NPY file not generated/found'})

    print("\n--- Aggregating and Plotting Metrics ---")
    if not all_metrics_data:
        print("No metrics data collected.")
        if os.path.exists(TEMP_OUTPUT_DIR):
             if not os.listdir(TEMP_OUTPUT_DIR):
                 os.rmdir(TEMP_OUTPUT_DIR)
        return

    successful_metrics = [m for m in all_metrics_data if isinstance(m, dict) and 'error' not in m]
    error_metrics = [m for m in all_metrics_data if isinstance(m, dict) and 'error' in m]

    if error_metrics:
        print("\nErrors occurred during processing for some models:")
        for m in error_metrics:
            print(f"- Model: {m.get('model_file', 'Unknown')}, Error: {m.get('error', 'Unknown')}")

    if not successful_metrics:
        print("\nNo successful evaluations to plot.")
        return

    metrics_df = pd.DataFrame(successful_metrics)
    metrics_df['model_id'] = metrics_df['model_file'].apply(extract_model_id)
    metrics_df['sortable_id'] = metrics_df['model_id'].apply(lambda x: pd.to_numeric(x, errors='ignore'))

    if pd.api.types.is_numeric_dtype(metrics_df['sortable_id']):
         metrics_df = metrics_df.sort_values(by='sortable_id')
    else:
         metrics_df['model_id'] = metrics_df['model_id'].astype(str)
         metrics_df = metrics_df.sort_values(by='model_id')
    metrics_df = metrics_df.drop(columns=['sortable_id'])

    print("\nMetrics DataFrame (Sorted):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(metrics_df.set_index('model_id').to_string())


    metrics_to_plot = [col for col in metrics_df.columns if col not in ['model_file', 'model_id'] and pd.api.types.is_numeric_dtype(metrics_df[col])]

    if not metrics_to_plot:
        print("\nNo numeric metrics found to plot.")
    else:
        print(f"\nPlotting {len(metrics_to_plot)} metrics...")
        num_metrics = len(metrics_to_plot)
        num_cols = min(3, num_metrics)
        num_rows = (num_metrics + num_cols - 1) // num_cols

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 7, num_rows * 5), squeeze=False)
        axes = axes.flatten()
        plot_x_axis = 'model_id'

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            plot_data = metrics_df.copy()
            plot_data[plot_x_axis] = plot_data[plot_x_axis].astype(str)

            sns.barplot(x=plot_x_axis, y=metric, data=plot_data, ax=ax, palette='viridis', dodge=False)
            ax.set_title(f'{metric}', fontsize=12)
            ax.set_xlabel('Model Identifier', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            if len(metrics_df[plot_x_axis].unique()) > 10:
                 ax.tick_params(axis='x', rotation=45, labelsize=8, ha='right')
            else:
                 ax.tick_params(axis='x', rotation=0, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            if not plot_data[metric].isnull().all():
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3g', fontsize=7, padding=3)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=2.0)
        plot_filename = "metrics_comparison_plot.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"\nSaved metrics comparison plot to {plot_filename}")
        plt.close(fig)

    cleanup = True
    if cleanup and os.path.exists(TEMP_OUTPUT_DIR):
        print(f"\nCleaning up temporary directory: {TEMP_OUTPUT_DIR}")
        shutil.rmtree(TEMP_OUTPUT_DIR)
        print("Temporary directory removed.")
    elif os.path.exists(TEMP_OUTPUT_DIR):
        print(f"\nTemporary directory {TEMP_OUTPUT_DIR} was not removed (cleanup=False or error during run).")


if __name__ == "__main__":
    main()
    print("\nScript execution finished.")

