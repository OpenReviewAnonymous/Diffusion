
import os
import sys
import json
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from univariate.diffwave import (
    N, T, DATA_PATH, BATCH_SIZE, load_data
)
from univariate.log_utils import setup_logger

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox

# True parameter values
true_mu = 0.0
true_phi = 0.95
true_omega = 0.05
true_alpha = 0.05
true_beta = 0.90

def calibrate_garch(seed=None, save_results=False, logger=None):
    """
    Calibrate AR(1)-GARCH(1,1) model using the specified random seed

    Args:
    - seed: random seed, if None then not set
    - save_results: whether to save results to file
    - logger: logger, if None then create a new one

    Returns:
    - parameter dictionary, including estimated mu, phi, omega, alpha, beta
    """
    # If logger is not provided, create one
    if logger is None:
        logger = setup_logger("calibrate_garch")
    
    # Set random seed (if provided)
    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Using random seed: {seed}")
    
    # Load data
    # data = np.load(DATA_PATH)[:10_000,:]
    # Use load_data function to get training data
    data_loader, scaler, _ = load_data()
    train_loader, _, _ = data_loader
    data = train_loader.dataset.tensors[0].numpy()[:10_000,:]
    
    # Data dimensions
    num_samples, series_length = data.shape
    logger.info(f"Data shape: {num_samples} samples, each sample length {series_length}")
    
    # Compute mu and phi for AR(1) model (fit each time series once)
    def fit_ar1_model(series):
        ar_model = AutoReg(series, lags=1)
        ar_fit = ar_model.fit()
        mu_estimate = ar_fit.params[0]  # Intercept (mu)
        phi_estimate = ar_fit.params[1]  # Autoregressive coefficient (phi)
        return mu_estimate, phi_estimate
    
    # Fit all time series
    mu_estimates = []
    phi_estimates = []
    
    for i in tqdm(range(num_samples), desc="Fitting AR(1) model"):
        mu, phi = fit_ar1_model(data[i])
        mu_estimates.append(mu)
        phi_estimates.append(phi)
    
    mu_estimates = np.array(mu_estimates)
    phi_estimates = np.array(phi_estimates)
    
    # Output AR(1) model estimation results
    avg_mu = np.mean(mu_estimates)
    avg_phi = np.mean(phi_estimates)
    logger.info(f"True mu: {true_mu}, Estimated mu: {avg_mu}")
    logger.info(f"True phi: {true_phi}, Estimated phi: {avg_phi}")
    
    # To fit the GARCH model, we need residuals. Calculate AR(1) residuals
    residuals = data - mu_estimates[:, None] - phi_estimates[:, None] * np.roll(data, 1, axis=1)
    residuals[:, 0] = 0  # The first item has no previous value, set to 0
    
    # Define and fit GARCH(1,1) model
    def fit_garch_model(residuals):
        garch_model = arch_model(residuals, vol='Garch', p=1, q=1, dist='normal')
        garch_fit = garch_model.fit(disp="off")
        return garch_fit
    
    # Fit GARCH model for each sample's residuals
    garch_results = []
    for i in tqdm(range(num_samples), desc="Fitting GARCH model"):
        garch_fit = fit_garch_model(residuals[i])
        garch_results.append(garch_fit)
    
    # Extract GARCH model parameters
    omega_estimates = np.array([result.params['omega'] for result in garch_results])
    alpha_estimates = np.array([result.params['alpha[1]'] for result in garch_results])
    beta_estimates = np.array([result.params['beta[1]'] for result in garch_results])
    
    # Output GARCH(1,1) model parameters
    avg_omega = np.mean(omega_estimates)
    avg_alpha = np.mean(alpha_estimates)
    avg_beta = np.mean(beta_estimates)
    logger.info(f"True omega: {true_omega}, Estimated omega: {avg_omega}")
    logger.info(f"True alpha: {true_alpha}, Estimated alpha: {avg_alpha}")
    logger.info(f"True beta: {true_beta}, Estimated beta: {avg_beta}")
    
    # Create parameter dictionary
    params = {
        'mu': float(avg_mu),
        'phi': float(avg_phi),
        'omega': float(avg_omega),
        'alpha': float(avg_alpha),
        'beta': float(avg_beta),
        'seed': seed
    }
    
    # Save results (if requested)
    if save_results and seed is not None:
        os.makedirs("results", exist_ok=True)
        
        # Save as JSON
        with open(f"results/params_seed_{seed}.json", 'w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"Parameters saved to results/params_seed_{seed}.json")
    
    return params

if __name__ == "__main__":
    # Create logger
    logger = setup_logger("calibrate_garch_main")
    logger.info("Start AR(1)-GARCH(1,1) model calibration")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calibrate AR(1)-GARCH(1,1) model')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    args = parser.parse_args()
    
    logger.info(f"Command line arguments: seed={args.seed}, save={args.save}")
    
    # Run calibration
    params = calibrate_garch(seed=args.seed, save_results=args.save, logger=logger)
    
    logger.info("Calibration complete, final parameters:")
    for key, value in params.items():
        if key != 'seed':
            logger.info(f"{key}: {value}")

