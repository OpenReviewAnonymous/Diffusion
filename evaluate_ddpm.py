import torch
import numpy as np
import os
import logging
from ddpm import (
    setup_parameters,
    inference,
    get_timestamp,
    load_data,
    N, T,
    DDPM
)

# Setup device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
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

# Evaluate standard DDPM
def evaluate_ddpm():
    # Step 0: Initialize configuration
    config = {
        'num_epochs': 500,  # Keep consistent with training configuration
        'T': T,
        'device': device,
        'time_stamp': get_timestamp()
    }
    
    # Setup logging
    logger = setup_logging(config['time_stamp'])
    logger.info(f">>> Step 0: Initialize, timestamp {config['time_stamp']}")
    
    # Step 1: Setup noise parameters
    config = setup_parameters(config)
    
    # Step 2: Load data
    data_loaders, scaler, test_size, config = load_data(config)
    logger.info(f"Data loading completed, test sample count: {test_size}")
    
    # Step 3: Load standard DDPM model
    model_path = "univariate/trained_models/best_model_ddpm_2025-05-13_05-50-01.pth"
    model = DDPM(T_steps=config['T']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f">>> Step 1: Loaded best model {model_path}")
    
    # Step 4: Execute inference
    N_test = int(0.12 * N)  # Use 12% of N as test samples
    logger.info(f">>> Step 2: Generate {N_test} samples")
    samples_ddpm = inference(model, N_test, config)
    
    # Step 5: Save results
    save_path = f"samples_ddpm_{config['time_stamp']}.npy"
    np.save(save_path, samples_ddpm)
    logger.info(f">>> Step 3: Samples saved to {save_path}")
    logger.info("Generated data is normalized data, use scaler if original scale is needed")

if __name__ == "__main__":
    evaluate_ddpm()
