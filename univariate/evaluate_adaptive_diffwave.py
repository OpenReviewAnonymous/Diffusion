import torch
import numpy as np
import logging
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from univariate.adaptive_diffwave import (
    setup_parameters, setup_model, inference,
    get_timestamp, N, T, device,
    AdaptiveDiffWaveModel
)

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set logging
def setup_logging(time_stamp):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"adaptive_diffwave_evaluation_{time_stamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("adaptive_diffwave_eval")

def evaluate_adaptive_diffwave(model_path):
    # Step 0: Initialize configuration
    config = {
        'num_epochs': 1250,  # Keep consistent with training configuration
        'T': T,
        'device': device,
        'time_stamp': get_timestamp()
    }
    
    # Set logging
    logger = setup_logging(config['time_stamp'])
    logger.info(f">>> Step 0: Initialization, timestamp {config['time_stamp']}")

    # Step 1: Set noise parameters
    config = setup_parameters(config)

    # Step 2: Load adaptive model
    logger.info(">>> Step 1: Set and load model")
    
    # Use single channel model
    model_adaptive = AdaptiveDiffWaveModel(T=T).to(device)
    logger.info(f"Attempting to load model: {model_path}")
    
    # Create model with same structure as saved model
    try:
        # Try to load model weights
        model_adaptive.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Successfully loaded model weights")
    except RuntimeError as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Please ensure model structure matches and use correct pretrained weight file")
        return
    
    model_adaptive.eval()
    logger.info(f">>> Step 1: Loaded best adaptive model {model_path}")

    # Step 3: Execute adaptive inference
    N_test = int(0.12 * N)
    logger.info(f">>> Step 2: Generate {N_test} adaptive samples")
    samples_adaptive = inference(
        model_adaptive,
        N_test,
        config['T'],
        config['alphas_dw'],
        config['alpha_bars_dw'],
        config['noise_schedule']
    )

    # Check generated sample shape
    logger.info(f"Generated sample shape: {samples_adaptive.shape}")
    
    # If sample shape is (N_test, T), reshape to (N_test, T, 1)
    if len(samples_adaptive.shape) == 2:
        samples_adaptive = samples_adaptive.reshape(N_test, T, 1)
        logger.info(f"Adjusted sample shape: {samples_adaptive.shape}")

    # Step 4: Save results
    save_path = f"samples_adaptive_diffwave_{config['time_stamp']}.npy"
    np.save(save_path, samples_adaptive)
    logger.info(f">>> Step 3: Adaptive samples saved to {save_path}")

    # Optional: Save Cholesky factor
    L_matrix = model_adaptive.get_L().detach().cpu().numpy()
    L_path = f"L_matrix_adaptive_diffwave_{config['time_stamp']}.npy"
    np.save(L_path, L_matrix)
    logger.info(f"Saved learned Cholesky factor to {L_path}")
    logger.info("Generated data is normalized data, if original scale needed, use scaler")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate adaptive DiffWave model")
    parser.add_argument("model_path", type=str, help="Path to model file")
    args = parser.parse_args()
    
    evaluate_adaptive_diffwave(args.model_path)
    
    
