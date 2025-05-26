import torch
import numpy as np
import logging
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from univariate.diffwave import (
    setup_parameters, setup_model, inference,
    get_timestamp, N, T, DiffWaveModel
)

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
def setup_logging(time_stamp):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"diffwave_evaluation_{time_stamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("diffwave_eval")

def evaluate_diffwave(model_path):
    # Step 0: Initialize configuration
    config = {
        'num_epochs': 500,  # Keep parameter consistency even though not needed for evaluation
        'T': T,
        'device': device,
        'time_stamp': get_timestamp()
    }
    
    # Set up logging
    logger = setup_logging(config['time_stamp'])
    logger.info(f">>> Step 0: Initialization, timestamp {config['time_stamp']}")

    # Step 1: Set noise parameters
    config = setup_parameters(config)

    # Step 2: Load model
    logger.info(">>> Step 1: Set up and load model")
    # model_diffwave, _, _, _ = setup_model(config['num_epochs'])
    # Modification: Directly create single-channel model
    model_diffwave = DiffWaveModel(T=T).to(device)
    logger.info(f"Trying to load model: {model_path}")
    
    model_diffwave.load_state_dict(torch.load(model_path, map_location=device))
    model_diffwave = model_diffwave.to(device)
    model_diffwave.eval()
    logger.info(f">>> Step 1: Loaded best model {model_path}")

    # Step 3: Run inference
    N_test = int(0.12 * N)  # Directly calculate 12% of N
    logger.info(f">>> Step 2: Generate {N_test} samples")
    samples_diffwave = inference(
        model_diffwave,
        N_test,
        config['T'],
        config['alphas_dw'],
        config['alpha_bars_dw'],
        config['noise_schedule']
    )

    # Check the shape of generated samples
    logger.info(f"Generated sample shape: {samples_diffwave.shape}")
    
    # If the sample shape is (N_test, T), reshape to (N_test, T, 1)
    if len(samples_diffwave.shape) == 2:
        samples_diffwave = samples_diffwave.reshape(N_test, T, 1)
        logger.info(f"Reshaped sample shape: {samples_diffwave.shape}")

    # Step 4: Save results
    save_path = f"samples_diffwave_{config['time_stamp']}.npy"
    np.save(save_path, samples_diffwave)
    logger.info(f">>> Step 3: Samples saved to {save_path}")
    logger.info("The generated data is normalized data. If you need the original scale, please use the scaler.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DiffWave model")
    parser.add_argument("model_path", type=str, help="Path to the model file")
    args = parser.parse_args()
    
    evaluate_diffwave(args.model_path)