
import torch
import numpy as np
import logging
from adaptive_diffwave import (
    setup_parameters, setup_model, inference,
    get_timestamp, N, T, device,
    AdaptiveDiffWaveModel
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
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

def evaluate_adaptive_diffwave():
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

    # Step 1: Setup noise parameters
    config = setup_parameters(config)

    # Step 2: Load adaptive model
    model_adaptive, _, _, _ = setup_model(config['num_epochs'])
    model_path = "univariate/trained_models/best_model_adaptive_diffwave_20250514_0511-23.pth"
    model_adaptive.load_state_dict(torch.load(model_path, map_location=device))
    model_adaptive = model_adaptive.to(device)
    model_adaptive.eval()
    logger.info(f">>> Step 1: Best adaptive model loaded {model_path}")

    # Step 3: Perform adaptive inference
    N_test = int(0.12 * N)
    logger.info(f">>> Step 2: Generating {N_test} adaptive samples")
    samples_adaptive = inference(
        model_adaptive,
        N_test,
        config['T'],
        config['alphas_dw'],
        config['alpha_bars_dw'],
        config['noise_schedule']
    )

    # Step 4: Save results
    save_path = f"samples_adaptive_diffwave_{config['time_stamp']}.npy"
    np.save(save_path, samples_adaptive)
    logger.info(f">>> Step 3: Adaptive samples saved to {save_path}")

    # Optional: Save Cholesky factor
    L_matrix = model_adaptive.get_L().detach().cpu().numpy()
    L_path = f"L_matrix_adaptive_diffwave_{config['time_stamp']}.npy"
    np.save(L_path, L_matrix)
    logger.info(f"Learned Cholesky factor saved to {L_path}")
    logger.info("Generated data is normalized data, if original scale is needed, use a scaler")

if __name__ == "__main__":
    evaluate_adaptive_diffwave()
