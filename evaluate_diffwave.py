import torch
import numpy as np
import logging
from diffwave import (
    setup_parameters, setup_model, inference,
    get_timestamp, N, T
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
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

def evaluate_diffwave():
    # Step 0: Initialize configuration
    config = {
        'num_epochs': 500,  # Although evaluation doesn't need it, keep parameter consistency
        'T': T,
        'device': device,
        'time_stamp': get_timestamp()
    }
    
    # Setup logging
    logger = setup_logging(config['time_stamp'])
    logger.info(f">>> Step 0: Initialize, timestamp {config['time_stamp']}")

    # Step 1: Setup noise parameters
    config = setup_parameters(config)

    # Step 2: Load model
    model_diffwave, _, _, _ = setup_model(config['num_epochs'])
    model_path = "univariate/trained_models/best_model_diffwave_2025-05-13_06:03.pth"
    model_diffwave.load_state_dict(torch.load(model_path, map_location=device))
    model_diffwave = model_diffwave.to(device)
    model_diffwave.eval()
    logger.info(f">>> Step 1: Loaded best model {model_path}")

    # Step 3: Execute inference
    N_test = int(0.12 * N)  # Calculate directly as 12% of N
    logger.info(f">>> Step 2: Generate {N_test} samples")
    samples_diffwave = inference(
        model_diffwave,
        N_test,
        config['T'],
        config['alphas_dw'],
        config['alpha_bars_dw'],
        config['noise_schedule']
    )

    # Step 4: Save results
    save_path = f"samples_diffwave_{config['time_stamp']}.npy"
    np.save(save_path, samples_diffwave)
    logger.info(f">>> Step 3: Samples saved to {save_path}")
    logger.info("Generated data is normalized data, use scaler if original scale is needed")

if __name__ == "__main__":
    evaluate_diffwave()