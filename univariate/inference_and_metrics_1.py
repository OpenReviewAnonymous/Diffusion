# -*- coding: utf-8 -*-
"""
Univariate time series generative model evaluation script
Performs inference and computes metrics comparing generated samples with real data, all outputs are logged to a log file
"""

import numpy as np
import os
import json
import logging
import sys # Import sys to access command-line arguments
from datetime import datetime
import pandas as pd  # Add pandas import

# Import metric calculation modules
from metrics4evaluation_1 import load_datasets, compare_datasets
# load_datasets is only used to load real data

# Add direct print statements to the console before logging configuration
print("Script started...")
print(f"Command line arguments: {sys.argv[1:]}")

# --- LOGGING SETUP ---
LOG_DIR = "logs" # Specifies the directory for log files
# Create the log directory if it doesn't exist.
# exist_ok=True prevents an error if the directory already exists.
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a timestamp string for unique log file naming.
# Format: YYYYMMDD_HHMMSS
LOG_FILE_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# Construct the full path for the log file.
LOG_FILE_PATH = os.path.join(LOG_DIR, f"inference_and_metrics_{LOG_FILE_TIMESTAMP}.log")

# Configure the logging system.
logging.basicConfig(
    level=logging.INFO, # Set the minimum logging level to INFO.
    # Define the format for log messages: timestamp, log level, and the message.
    format="%(asctime)s - %(levelname)s - %(message)s",
    # Specify handlers for logging output. Here, it's a file handler.
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        # Removed logging.StreamHandler() to ensure no console output as per instructions
    ]
)
# --- END LOGGING SETUP ---

def process_generated_data(real_data: np.ndarray, generated_samples_path: str, dataset_name: str) -> dict:
    """
    Load generated univariate time series data, compare it with real data, and return computed evaluation metrics

    Args:
        real_data: Real dataset (NumPy array), shape (N, T, 1)
        generated_samples_path: Path to the generated samples .npy file
        dataset_name: Descriptive name of the generated dataset (for logging/reporting)

    Returns:
        Dictionary containing computed metrics or error message
    """
    # Validate the generated samples path
    if not generated_samples_path or not os.path.exists(generated_samples_path):
        logging.error(f"Generated samples path is invalid or file does not exist for {dataset_name}: {generated_samples_path}")
        return {"error": f"Invalid generated samples path for {dataset_name}"}

    # Load generated data from .npy file
    # As required, do not use try-except block
    generated_data = np.load(generated_samples_path)
    if generated_data is None or generated_data.size == 0:
        logging.error(f"Generated data from {generated_samples_path} ({dataset_name}) is empty.")
        return {"error": f"Generated data is empty for {dataset_name}"}

    # Ensure generated data is in 3D format (N,T,1)
    if generated_data.ndim == 2:  # If in (N,T) format
        logging.info(f"Converting 2D data {generated_data.shape} to 3D format (N,T,1)")
        generated_data = generated_data.reshape(generated_data.shape[0], generated_data.shape[1], 1)
    elif generated_data.ndim == 3 and generated_data.shape[2] > 1:  # If multivariate (N,T,D), D>1
        logging.info(f"Extracting first dimension from multivariate data {generated_data.shape}")
        generated_data = generated_data[:, :, 0:1]  # Keep only the first variable

    # Ensure real_data is also in 3D format
    if real_data.ndim == 2:
        real_data = real_data.reshape(real_data.shape[0], real_data.shape[1], 1)
    elif real_data.ndim == 3 and real_data.shape[2] > 1:
        real_data = real_data[:, :, 0:1]

    # Check data shapes, log warning if mismatched
    if real_data.shape[1:] != generated_data.shape[1:]:
        logging.warning(
            f"Shape mismatch (beyond batch size): Real data {real_data.shape}, "
            f"Generated data {dataset_name} {generated_data.shape}."
        )
        # Continue calculation despite warning

    # Compute metrics, compare real and generated data
    logging.info(f"Comparing datasets for {dataset_name}...")
    metrics_results = compare_datasets(real_data, generated_data, dataset_name)

    # Convert metrics to JSON serializable types (e.g., Python float)
    printable_metrics = {}
    for key, value in metrics_results.items():
        if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            if not np.isnan(value): # Ensure NaN values are not included directly
                printable_metrics[key] = float(value)
        elif value is not None and not (isinstance(value, float) and np.isnan(value)):
            printable_metrics[key] = value
    
    # Log wd_value1 and wd_value2 values
    if 'wd_value1' in printable_metrics and 'wd_value2' in printable_metrics:
        logging.info(f"{dataset_name} - wd_value1: {printable_metrics['wd_value1']:.6f}, wd_value2: {printable_metrics['wd_value2']:.6f}")
    
    logging.info(f"Metrics calculation completed for {dataset_name}.")
    return printable_metrics

def summarize_results_as_dataframe(all_metrics_results: dict) -> pd.DataFrame:
    """
    Summarize all model evaluation results into a DataFrame and return it
    
    Args:
        all_metrics_results: Dictionary containing all model evaluation results, format {model_name: {metric_name: value}}
        
    Returns:
        pandas.DataFrame: DataFrame containing all evaluation metrics, rows are model names, columns are metric names
    """
    # Prepare an empty DataFrame
    models = list(all_metrics_results.keys())
    
    # Get all possible metric names (merge all model metric names)
    all_metrics = set()
    for model_metrics in all_metrics_results.values():
        all_metrics.update(model_metrics.keys())
    
    # Remove 'error' if present
    if 'error' in all_metrics:
        all_metrics.remove('error')
    
    # Create data dictionary
    data = {}
    for metric in all_metrics:
        data[metric] = []
        for model in models:
            model_metrics = all_metrics_results[model]
            if metric in model_metrics and not (isinstance(model_metrics[metric], float) and np.isnan(model_metrics[metric])):
                data[metric].append(model_metrics[metric])
            else:
                data[metric].append(np.nan)
    
    # Create DataFrame
    df = pd.DataFrame(data, index=models)
    
    return df

def main():
    """
    Main function: coordinates data loading, metric calculation, and logging
    Processes .npy files provided as command-line arguments
    """
    print("Entering main function...")
    logging.info("Starting inference and metrics calculation script.")

    # Get generated sample paths from command-line arguments
    # The first argument sys.argv[0] is the script name itself, so slice from index 1
    generated_sample_paths = sys.argv[1:]
    print(f"Received arguments: {generated_sample_paths}")

    if not generated_sample_paths:
        print("Error: No generated sample paths provided as command-line arguments. Exiting.")
        logging.error("No generated sample paths provided as command-line arguments. Exiting.")
        return

    # Load real dataset
    print("Starting to load real dataset...")
    logging.info("Loading real dataset...")
    real_data_tuple = load_datasets() # Assume load_datasets returns a tuple or similar structure

    real_data = None
    if real_data_tuple and len(real_data_tuple) > 0:
        real_data = real_data_tuple[0] # Assume the first element is the required dataset

    if real_data is None or (isinstance(real_data, np.ndarray) and real_data.size == 0):
        print("Error: Failed to load real data or real data is empty. Exiting.")
        logging.error("Failed to load real data or real data is empty. Exiting.")
        return

    # Ensure real_data is a NumPy array
    if not isinstance(real_data, np.ndarray):
        real_data = np.array(real_data) # Convert if not ndarray

    # Ensure real_data is in 3D format (N,T,1)
    if real_data.ndim == 2:  # If in (N,T) format
        print(f"Converting real data from 2D shape {real_data.shape} to 3D format")
        logging.info(f"Converting real data from 2D shape {real_data.shape} to 3D (N,T,1) format")
        real_data = real_data.reshape(real_data.shape[0], real_data.shape[1], 1)
    elif real_data.ndim == 3 and real_data.shape[2] > 1:  # If multivariate (N,T,D), D>1
        print(f"Extracting first dimension from multivariate real data {real_data.shape}")
        logging.info(f"Extracting first dimension from multivariate real data {real_data.shape}")
        real_data = real_data[:, :, 0:1]  # Keep only the first variable

    print(f"Real data loaded successfully. Shape: {real_data.shape}")
    logging.info(f"Real data loaded successfully. Shape: {real_data.shape}")

    all_metrics_results = {} # Dictionary to store results for all datasets

    # Process each specified .npy file
    for full_file_path in generated_sample_paths:
        print(f"Processing file: {full_file_path}")
        if not os.path.isfile(full_file_path):
            print(f"Error: Provided path is not a file or does not exist: {full_file_path}. Skipping.")
            logging.error(f"Provided path is not a file or does not exist: {full_file_path}. Skipping.")
            continue

        # Use the file name as the dataset name for clear identification
        dataset_name = os.path.basename(full_file_path)

        logging.info(f"Processing file: {full_file_path} (Dataset Name: {dataset_name})")
        # Call the processing function for each individual .npy file
        metrics = process_generated_data(real_data, full_file_path, dataset_name)

        if 'error' in metrics:
            logging.error(f"Error processing file {full_file_path}: {metrics['error']}")

        all_metrics_results[dataset_name] = metrics # Store results using dataset name as key

    # Log the merged metrics results in JSON format
    logging.info("====== ALL METRICS RESULTS ======")
    # Convert metrics dictionary to a well-formatted JSON string
    # If data is not serializable, may raise error; as instructed, do not use try-except
    metrics_json = json.dumps(all_metrics_results, indent=4, ensure_ascii=False)
    # Log JSON string line by line for better readability in log file
    for line in metrics_json.splitlines():
        logging.info(line)
    logging.info("====== END ALL METRICS RESULTS ======")
    
    # Summarize results as DataFrame and log to file
    results_df = summarize_results_as_dataframe(all_metrics_results)
    
    # Print DataFrame to console
    print("\n====== Evaluation Results Summary (DataFrame) ======")
    print(results_df.to_string())
    
    # Log DataFrame to file
    logging.info("====== EVALUATION RESULTS SUMMARY (DATAFRAME) ======")
    for line in results_df.to_string().splitlines():
        logging.info(line)
    logging.info("====== END EVALUATION RESULTS SUMMARY ======")
    
    print("Script finished.")
    logging.info("Script finished.")

if __name__ == "__main__":
    main()
