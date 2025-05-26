import torch
import numpy as np
from adaptiv_ddpm_regularity import ( # Changed import source
    setup_parameters,
    inference,
    get_timestamp,
    load_data,
    N, T,
    AdaptiveDDPM
)
import os
import logging # Added for logging

# Set device
# It's good practice to set this early, similar to the training script
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Or manage externally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup basic logging for the evaluation script
eval_log_dir = "logs_eval"
os.makedirs(eval_log_dir, exist_ok=True)
eval_run_id = get_timestamp() # Unique ID for this evaluation run
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(eval_log_dir, f"evaluate_regularity_{eval_run_id}.log")),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Function to run evaluation for a given model
def run_evaluation(model_path: str, output_dir: str) -> str:
    """
    Runs inference using a trained AdaptiveDDPM model from adaptiv_ddpm_regularity.py
    and saves the generated samples.

    Args:
        model_path: Path to the pre-trained model (.pth file).
        output_dir: Directory to save the generated samples (.npy file).

    Returns:
        Path to the saved .npy file containing generated samples.
    """
    eval_timestamp_func = get_timestamp() # Timestamp for this specific evaluation
    config = {
        # 'num_epochs': 1250, # Not needed for inference, but T is
        'T': T,
        'device': device,
        'time_stamp': eval_timestamp_func, # For any internal use in imported functions
        # 'run_id': model_path.split('/')[-1].split('.')[0] # Attempt to get run_id from model path if needed
    }
    logger.info(f">>> Step 0: Initializing evaluation for {os.path.basename(model_path)} with run ID {eval_run_id}")
    logger.info(f"Using device: {device}")

    logger.info(">>> Step 1: Setting up parameters")
    config = setup_parameters(config) # betas, alphas, alpha_bars

    logger.info(">>> Step 2: Loading data (for scalers and test_size)")
    # We might not need the full data_loader, but load_data usually returns scalers and test_size
    _, scalers, test_size_from_data = load_data() # Assuming load_data from adaptiv_ddpm_regularity

    logger.info(">>> Step 3: Initializing and loading model")
    model_adaptive = AdaptiveDDPM(T=T, D=2).to(device) # D=2 as per adaptiv_ddpm_regularity
    model_adaptive.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Successfully loaded model from {model_path}")

    model_adaptive.eval()

    # Determine N_test, similar to the example evaluation script
    # You might want to make N_test configurable or use a fixed portion
    N_test = int(0.12 * N) # Using the same fraction as the example
    logger.info(f">>> Step 4: Generating {N_test} adaptive samples")
    
    samples_adaptive = inference(
        model_adaptive,
        N_test,
        scalers, # Pass scalers if your inference function uses them (e.g., for inverse transform)
        config   # Pass config for betas, alphas etc.
    )

    logger.info(">>> Step 5: Saving generated samples")
    base_model_name = os.path.basename(model_path).replace(".pth", "")
    # Include eval_run_id in the filename for uniqueness if multiple evals are run for the same model
    save_filename = f"samples_{base_model_name}_eval_{eval_run_id}.npy"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_filename)
    
    np.save(save_path, samples_adaptive)
    logger.info(f"Saved adaptive samples to {save_path}")

    return save_path

if __name__ == "__main__":
    logger.info("Starting evaluation script in standalone mode.")
    # Example usage:
    # IMPORTANT: Replace with the actual path to your trained model
    # from adaptiv_ddpm_regularity.py
    # The model path should correspond to a .pth file saved during the training
    # of adaptiv_ddpm_regularity.py, e.g., "best_model_adaptive_ddpm_YOUR_RUN_ID.pth"
    
    # Try to get model_path from environment variable or use a default
    default_model_path = os.environ.get("EVAL_MODEL_PATH_REGULARITY", "/logs/best_model_adaptive_ddpm_lambda1_1e-2_lambda2_1e-2.pth")
    default_output_dir = "evaluation_output_regularity"

    if not os.path.exists(default_model_path) or "REGULARITY_RUN_ID_EXAMPLE" in default_model_path:
        logger.warning(f"Default model path for standalone execution not found or is a placeholder: {default_model_path}")
        logger.warning("Please provide a valid model path by setting the EVAL_MODEL_PATH_REGULARITY environment variable or modifying the script.")
    else:
        logger.info(f"Attempting to evaluate model: {default_model_path}")
        logger.info(f"Output will be saved to: {default_output_dir}")
        generated_file_path = run_evaluation(model_path=default_model_path, output_dir=default_output_dir)
        logger.info(f"Standalone evaluation complete. Samples saved to: {generated_file_path}")
