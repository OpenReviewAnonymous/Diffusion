import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from univariate.diffwave import (
    get_timestamp
)

import numpy as np
from tqdm import tqdm
import argparse
import json
from univariate.log_utils import setup_logger

def generate_ar1_garch_series(T, mu=-0.0018539114395484581, phi=0.903381214425834, omega=0.2124798935755303, alpha=0.051978618406107534, beta=0.7229209775966785):
# def generate_ar1_garch_series(T, mu=0.0, phi=0.8, omega=0.05, alpha=0.05, beta=0.90):
    """
    Generate a time series using an AR(1) process with GARCH(1,1) volatility.
    
    Parameters:
    - T: int, the length of the time series.
    - mu: float, the mean of the AR(1) process.
    - phi: float, the autoregressive coefficient.
    - omega: float, the constant term in the GARCH model.
    - alpha: float, the coefficient for the lagged squared error in GARCH.
    - beta: float, the coefficient for the lagged variance in GARCH.
    
    Returns:
    - x: numpy array of shape (T,), the generated time series.
    """
    # Initialize arrays for conditional variance (sigma2), error term (eps), and the time series (x)
    sigma2 = np.zeros(T)
    eps = np.zeros(T)
    x = np.zeros(T)
    
    # Initialization: sigma_0^2 = omega / (1 - alpha - beta)
    sigma2[0] = omega / (1 - alpha - beta)
    # Generate the initial error and time series value
    eps[0] = np.sqrt(sigma2[0]) * np.random.randn()
    x[0] = mu + eps[0]
    
    # Recursively generate the series
    for t in range(1, T):
        sigma2[t] = omega + alpha * (eps[t-1] ** 2) + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * np.random.randn()
        x[t] = mu + phi * x[t-1] + eps[t]
        
    return x

def generate_dataset(params, seed=None, num_samples=10000, series_length=100, output_file=None, logger=None):
    """
    Generate an AR(1)-GARCH(1,1) time series dataset.
    
    Parameters:
    - params: a dictionary containing mu, phi, omega, alpha, beta
    - seed: random seed
    - num_samples: number of samples to generate
    - series_length: length of each time series
    - output_file: output file name, if None use default
    - logger: logger, if None create a new one
    
    Returns:
    - saved file name
    """
    # If logger is not provided, create one
    if logger is None:
        logger = setup_logger("evaluate_garch")
    
    # Set random seed (if provided)
    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Using random seed: {seed}")
    
    # Extract parameters
    mu = params['mu']
    phi = params['phi']
    omega = params['omega']
    alpha = params['alpha']
    beta = params['beta']
    
    logger.info(f"Using parameters: mu={mu}, phi={phi}, omega={omega}, alpha={alpha}, beta={beta}")
    
    # Generate dataset: each row is a time series sample
    data = np.array([
        generate_ar1_garch_series(series_length, mu, phi, omega, alpha, beta) 
        for _ in tqdm(range(num_samples), desc="Generating samples")
    ])
    
    # Set output file name
    if (output_file is None) and (seed is not None):
        output_file = f"calibrated_ar1_garch1_1_inferenced_seed_{seed}_{get_timestamp()}.npy"
    
    # Save generated data
    np.save(output_file, data)
    logger.info(f"Saved {num_samples} samples to {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Create logger
    logger = setup_logger("evaluate_garch_main")
    logger.info("Start generating samples")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate AR(1)-GARCH(1,1) time series')
    parser.add_argument('--params_file', type=str, help='JSON file containing parameters')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--series_length', type=int, default=100, help='Length of each time series')
    parser.add_argument('--output_file', type=str, help='Output file name')
    parser.add_argument('--mu', type=float, help='mu parameter of AR(1) model')
    parser.add_argument('--phi', type=float, help='phi parameter of AR(1) model')
    parser.add_argument('--omega', type=float, help='omega parameter of GARCH model')
    parser.add_argument('--alpha', type=float, help='alpha parameter of GARCH model')
    parser.add_argument('--beta', type=float, help='beta parameter of GARCH model')
    args = parser.parse_args()
    
    logger.info(f"Command line arguments: params_file={args.params_file}, seed={args.seed}, " +
                f"num_samples={args.num_samples}, series_length={args.series_length}")
    
    # Get parameters
    if args.params_file:
        # Load parameters from file
        with open(args.params_file, 'r') as f:
            params = json.load(f)
        logger.info(f"Loaded parameters from file: {args.params_file}")
    else:
        # Use command line parameters
        params = {
            'mu': args.mu if args.mu is not None else -0.0018539114395484581,
            'phi': args.phi if args.phi is not None else 0.903381214425834,
            'omega': args.omega if args.omega is not None else 0.2124798935755303,
            'alpha': args.alpha if args.alpha is not None else 0.051978618406107534,
            'beta': args.beta if args.beta is not None else 0.7229209775966785
        }
        logger.info("Using command line or default parameters")
    
    # Generate dataset
    output_file = generate_dataset(
        params, 
        seed=args.seed, 
        num_samples=args.num_samples, 
        series_length=args.series_length,
        output_file=args.output_file,
        logger=logger
    )
    
    logger.info(f"Sample generation completed, saved to {output_file}")
