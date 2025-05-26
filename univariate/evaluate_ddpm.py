import torch
import numpy as np
import os
import logging
import sys
import argparse
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from univariate.ddpm import (
    setup_parameters,
    inference,
    get_timestamp,
    load_data,
    N, T,
    DDPM
)

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
def setup_logging(time_stamp):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ddpm_evaluation_{time_stamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ddpm_eval")

# Extract timestamp from model path
def extract_timestamp_from_model_path(model_path):
    # Extract timestamp from filename
    filename = os.path.basename(model_path)
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
    
    return timestamp_match.group(1)


# Evaluate standard DDPM
def evaluate_ddpm(model_path):
    # Step 0: Initialize configuration
    config = {
        'num_epochs': 500,  # Keep consistent with training configuration
        'T': T,
        'device': device,
        'time_stamp': extract_timestamp_from_model_path(model_path)
    }
    
    # Set up logging
    logger = setup_logging(config['time_stamp'])
    logger.info(f">>> Step 0: Initialization, using model timestamp {config['time_stamp']}")
    
    # Step 1: Set noise parameters
    config = setup_parameters(config)
    
    # Step 2: Load data
    data_loaders, scaler, test_size, config = load_data(config)
    logger.info(f"Data loaded, number of test samples: {test_size}")
    
    # Step 3: Load standard DDPM model
    logger.info(f"Trying to load model: {model_path}")
    
    model = DDPM(T_steps=config['T']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f">>> Step 1: Loaded best model {model_path}")
    
    # Step 4: Perform inference
    N_test = int(0.12 * N)  # Use 12% of N as test samples
    logger.info(f">>> Step 2: Generating {N_test} samples")
    samples_ddpm = inference(model, N_test, config)
    
    # Check the shape of generated samples
    logger.info(f"Generated sample shape: {samples_ddpm.shape}")
    
    # If the sample shape is (N_test, T), reshape to (N_test, T, 1)
    if len(samples_ddpm.shape) == 2:
        samples_ddpm = samples_ddpm.reshape(N_test, T, 1)
        logger.info(f"Reshaped sample shape: {samples_ddpm.shape}")
    
    # Step 5: Save results
    save_path = f"samples_ddpm_{config['time_stamp']}.npy"
    np.save(save_path, samples_ddpm)
    logger.info(f">>> Step 3: Samples saved to {save_path}")
    logger.info("The generated data is normalized data. If you need the original scale, please use the scaler.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DDPM model")
    parser.add_argument("--model_path", type=str, help="Path to the model file", default="best_model_ddpm_2025-05-15_05-04-29.pth")
    args = parser.parse_args()
    
    evaluate_ddpm(args.model_path)
