import torch
import numpy as np
import logging
import sys
import os
import re
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from univariate.adaptive_ddpm import (
    setup_parameters,
    inference,
    get_timestamp,
    load_data,
    N, T,
    AdaptiveDDPM
)

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
def setup_logging(time_stamp):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"adaptive_ddpm_evaluation_{time_stamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("adaptive_ddpm_eval")


# Extract timestamp from model path
def extract_timestamp_from_model_path(model_path):
    # Extract timestamp from filename
    print(f"model_path: {model_path}")
    filename = os.path.basename(model_path)
    print(f"filename: {filename}")
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}:\d{2})', filename)
    print(f"timestamp_match: {timestamp_match}")
    return timestamp_match.group(1)

# Evaluate Adaptive DDPM
def evaluate_adaptive_ddpm(model_path):
    # Step 0: Initialize config
    config = {
        'num_epochs': 1250,  # Keep consistent with training config
        'T': T,
        'device': device,
        'time_stamp': extract_timestamp_from_model_path(model_path)
    }
    
    # Set up logging
    logger = setup_logging(config['time_stamp'])
    logger.info(f">>> Step 0: Initialization, timestamp {config['time_stamp']}")
    
    # Step 1: Set noise parameters
    config = setup_parameters(config)
    
    # Step 2: Load data
    data_loader, scaler, _ = load_data()
    logger.info("Data loading completed")

    # Step 3: Load adaptive model
    logger.info(f"Trying to load model: {model_path}")
    
    # Modification: Use correct initialization arguments - AdaptiveDDPM does not accept D argument
    model_adaptive = AdaptiveDDPM(T=T).to(device)
    model_adaptive.load_state_dict(torch.load(model_path, map_location=device))
    model_adaptive.eval()
    logger.info(f">>> Step 1: Loaded best adaptive model {model_path}")

    # Step 4: Perform adaptive inference
    N_test = int(0.12 * N)
    logger.info(f">>> Step 2: Generating {N_test} adaptive samples")
    samples_adaptive = inference(
        model_adaptive,
        N_test,
        scaler,
        config
    )

    # Check generated sample shape
    logger.info(f"Generated sample shape: {samples_adaptive.shape}")
    
    # If sample shape is (N_test, T), reshape to (N_test, T, 1)
    if len(samples_adaptive.shape) == 2:
        samples_adaptive = samples_adaptive.reshape(N_test, T, 1)
        logger.info(f"Reshaped sample shape: {samples_adaptive.shape}")

    # Step 5: Save results
    save_path = f"samples_adaptive_ddpm_{config['time_stamp']}.npy"
    np.save(save_path, samples_adaptive)
    logger.info(f">>> Step 3: Adaptive samples saved to {save_path}")

    # Optional: Save Cholesky factor
    L_matrix = model_adaptive.get_L().detach().cpu().numpy()
    L_path = f"L_matrix_adaptive_ddpm_{config['time_stamp']}.npy"
    np.save(L_path, L_matrix)
    logger.info(f"Saved learned Cholesky factor to {L_path}")
    logger.info("The generated data is normalized data. If you need the original scale, please use the scaler.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Adaptive DDPM model")
    parser.add_argument("--model_path", type=str, help="Path to model file", default="best_model_adaptive_ddpm_2025-05-15_04-50:23.pth")
    args = parser.parse_args()
    
    evaluate_adaptive_ddpm(args.model_path)