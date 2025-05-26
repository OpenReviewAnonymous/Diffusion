from tkinter import _test
import torch
import numpy as np
import logging
from adaptive_ddpm import (
    setup_parameters,  # Assuming setup_parameters is defined
    inference,  # Assuming inference is defined
    get_timestamp,  # Assuming get_timestamp is defined
    load_data,
    N, T,
    AdaptiveDDPM
)
import os

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
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

# Function to run evaluation for a given model
def run_evaluation(model_path: str, output_dir: str) -> str:
    """
    Runs inference using a trained AdaptiveDDPM model and saves the generated samples.

    Args:
        model_path: Path to the pre-trained model (.pth file).
        output_dir: Directory to save the generated samples (.npy file).

    Returns:
        Path to the saved .npy file containing generated samples.
    """
    eval_timestamp = get_timestamp()
    config = {
        'num_epochs': 1250,
        'T': T,
        'device': device,
        'time_stamp': eval_timestamp
    }
    print(f">>> Step 0: Initializing evaluation for {os.path.basename(model_path)}")

    config = setup_parameters(config)

    data_loader, scalers, _ = load_data()

    model_adaptive = AdaptiveDDPM(T=T, D=2).to(device)
    model_adaptive.load_state_dict(torch.load(model_path, map_location=device))

    model_adaptive.eval()
    print(f">>> Step 1: Loaded adaptive model from {model_path}")

    N_test = int(0.12 * N)
    print(f">>> Step 2: Generating {N_test} adaptive samples")
    samples_adaptive = inference(
        model_adaptive,
        N_test,
        scalers,
        config
    )

    base_model_name = os.path.basename(model_path).replace(".pth", "")
    save_filename = f"samples_{base_model_name}_{eval_timestamp}.npy"
    save_path = os.path.join(output_dir, save_filename)
    os.makedirs(output_dir, exist_ok=True)
    np.save(save_path, samples_adaptive)
    print(f">>> Step 3: Saved adaptive samples to {save_path}")

    return save_path

# Original main execution block (optional, can be kept for standalone runs)
if __name__ == "__main__":
    # Example usage if run directly
    # You would need to specify a model path and output directory
    default_model_path = "/logs/best_model_adaptive_ddpm_20250513_051955.pth" # Example path
    default_output_dir = "evaluation_output_adaptive_ddpm_2"
    # default_output_dir = "temp_evaluation_output_standalone"

    if not os.path.exists(default_model_path):
         print(f"Default model path for standalone execution not found: {default_model_path}")
         print("Please provide a valid model path or modify the script.")
    else:
         generated_file = run_evaluation(model_path=default_model_path, output_dir=default_output_dir)
         print(f"\nStandalone evaluation complete. Samples saved to: {generated_file}")

# Evaluate Adaptive DDPM
def evaluate_adaptive_ddpm():
    # Step 0: Initialize configuration
    config = {
        'num_epochs': 1250,  # Keep consistent with training configuration
        'T': T,
        'device': device,
        'time_stamp': get_timestamp()
    }
    
    # Setup logging
    logger = setup_logging(config['time_stamp'])
    logger.info(f">>> Step 0: Initialization, timestamp {config['time_stamp']}")
    
    # Step 1: Set noise parameters
    config = setup_parameters(config)
    
    # Step 2: Load data
    data_loader, scaler, _ = load_data()
    logger.info("Data loading complete")

    # Step 3: Load adaptive model
    model_path = "univariate/trained_models/best_model_adaptive_ddpm_2025-05-13_05-50.pth"
    model_adaptive = AdaptiveDDPM(T=T).to(device)
    model_adaptive.load_state_dict(torch.load(model_path, map_location=device))
    model_adaptive.eval()
    logger.info(f">>> Step 1: Best adaptive model loaded from {model_path}")

    # Step 4: Perform adaptive inference
    N_test = int(0.12 * N)
    logger.info(f">>> Step 2: Generating {N_test} adaptive samples")
    samples_adaptive = inference(
        model_adaptive,
        N_test,
        scaler,
        config
    )

    # Step 5: Save results
    save_path = f"samples_adaptive_ddpm_{config['time_stamp']}.npy"
    np.save(save_path, samples_adaptive)
    logger.info(f">>> Step 3: Adaptive samples saved to {save_path}")

    # Optional: Save Cholesky factor
    L_matrix = model_adaptive.get_L().detach().cpu().numpy()
    L_path = f"L_matrix_adaptive_ddpm_{config['time_stamp']}.npy"
    np.save(L_path, L_matrix)
    logger.info(f"Learned Cholesky factor saved to {L_path}")
    logger.info("Generated data is normalized data, use scaler if original scale is needed")

if __name__ == "__main__":
    evaluate_adaptive_ddpm()