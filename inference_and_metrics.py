# -*- coding: utf-8 -*-
"""
Script to perform inference and calculate metrics for two sets of generated samples
compared against real data. All outputs are logged to a file.
"""

import numpy as np
import os
import json
import logging
import sys # Import sys to access command-line arguments
from datetime import datetime

# Ensure 'metrics4evaluation.py' is accessible, e.g., in the same directory or PYTHONPATH
# We assume it provides load_datasets and compare_datasets functions.
from metrics4evaluation import load_datasets, compare_datasets

# Add direct console print statements before logging configuration
print("Script starting execution...")
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
    Loads generated data, compares it with real data, and returns metrics.

    Args:
        real_data: The real dataset (as a NumPy array).
        generated_samples_path: Path to the .npy file for the generated samples.
        dataset_name: A descriptive name for the generated dataset (for logging/reporting).

    Returns:
        A dictionary containing calculated metrics or an error message.
    """
    # Validate the path for generated samples.
    if not generated_samples_path or not os.path.exists(generated_samples_path):
        logging.error(f"Generated samples path is invalid or file does not exist for {dataset_name}: {generated_samples_path}")
        return {"error": f"Invalid generated samples path for {dataset_name}"}

    # Load the generated data from the .npy file.
    # No try-except block here as per requirements.
    generated_data = np.load(generated_samples_path)
    if generated_data is None or generated_data.size == 0:
        logging.error(f"Generated data from {generated_samples_path} ({dataset_name}) is empty.")
        return {"error": f"Generated data is empty for {dataset_name}"}

    # Log a warning if data shapes (excluding batch size) do not match.
    if real_data.shape[1:] != generated_data.shape[1:]:
        logging.warning(
            f"Shape mismatch (beyond batch size): Real data {real_data.shape}, "
            f"Generated data {dataset_name} {generated_data.shape}."
        )
        # Continue with calculations despite the warning.

    # Calculate metrics by comparing real and generated data.
    logging.info(f"Comparing datasets for {dataset_name}...")
    metrics_results = compare_datasets(real_data, generated_data, dataset_name)

    # Convert metrics to JSON-serializable types (e.g., Python floats).
    printable_metrics = {}
    for key, value in metrics_results.items():
        if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            if not np.isnan(value): # Ensure NaN values are not included directly
                printable_metrics[key] = float(value)
        elif value is not None and not (isinstance(value, float) and np.isnan(value)):
            printable_metrics[key] = value
    
    # Record wd_value1 and wd_value2 values
    if 'wd_value1' in printable_metrics and 'wd_value2' in printable_metrics:
        logging.info(f"{dataset_name} - wd_value1: {printable_metrics['wd_value1']:.6f}, wd_value2: {printable_metrics['wd_value2']:.6f}")
    
    logging.info(f"Metrics calculation completed for {dataset_name}.")
    return printable_metrics

def main():
    """
    Main function to orchestrate data loading, metric calculation, and logging.
    Processes .npy files provided as command-line arguments.
    """
    print("Entering main function...")
    logging.info("Starting inference and metrics calculation script.")

    # Get generated sample paths from command-line arguments
    # The first argument sys.argv[0] is the script name itself, so we slice from index 1
    generated_sample_paths = sys.argv[1:]
    print(f"Received parameters: {generated_sample_paths}")

    if not generated_sample_paths:
        print("Error: No generated sample paths provided as command-line arguments. Exiting.")
        logging.error("No generated sample paths provided as command-line arguments. Exiting.")
        return

    # Load the real dataset.
    print("Starting to load real dataset...")
    logging.info("Loading real dataset...")
    real_data_tuple = load_datasets() # Assuming load_datasets returns a tuple or similar

    real_data = None
    if real_data_tuple and len(real_data_tuple) > 0:
        real_data = real_data_tuple[0] # Assuming the first element is the desired dataset

    if real_data is None or (isinstance(real_data, np.ndarray) and real_data.size == 0):
        print("Error: Failed to load real data or real data is empty. Exiting.")
        logging.error("Failed to load real data or real data is empty. Exiting.")
        return

    # Ensure real_data is a NumPy array.
    if not isinstance(real_data, np.ndarray):
        real_data = np.array(real_data) # Convert if not already an ndarray.

    print(f"Real data loaded successfully. Shape: {real_data.shape}")
    logging.info(f"Real data loaded successfully. Shape: {real_data.shape}")

    all_metrics_results = {} # Dictionary to store results for all datasets.

    # Process each specified .npy file.
    for full_file_path in generated_sample_paths:
        print(f"Processing file: {full_file_path}")
        if not os.path.isfile(full_file_path):
            print(f"Error: Provided path is not a file or does not exist: {full_file_path}. Skipping.")
            logging.error(f"Provided path is not a file or does not exist: {full_file_path}. Skipping.")
            continue

        # Use the file name as the dataset name for clarity
        dataset_name = os.path.basename(full_file_path)

        logging.info(f"Processing file: {full_file_path} (Dataset Name: {dataset_name})")
        # Call the processing function for each individual .npy file.
        metrics = process_generated_data(real_data, full_file_path, dataset_name)

        if 'error' in metrics:
            logging.error(f"Error processing file {full_file_path}: {metrics['error']}")

        all_metrics_results[dataset_name] = metrics # Store results using the dataset name as key

    # Log the consolidated metrics results in JSON format.
    logging.info("====== ALL METRICS RESULTS ======")
    # Convert the dictionary of metrics to a pretty-printed JSON string.
    # This can raise an error if data is not serializable; per instructions, no try-except.
    metrics_json = json.dumps(all_metrics_results, indent=4, ensure_ascii=False)
    # Log each line of the JSON string for better readability in log files.
    for line in metrics_json.splitlines():
        logging.info(line)
    logging.info("====== END ALL METRICS RESULTS ======")
    print("Script execution completed.")
    logging.info("Script finished.")

if __name__ == "__main__":
    main()
