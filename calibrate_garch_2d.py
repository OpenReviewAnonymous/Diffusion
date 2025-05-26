import numpy as np
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

# Add relative path to allow importing the diffwave module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffwave import load_data, inverse_transform_channels

# ---------------------------------- #A10
import torch

# Set device: use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[#A10] Using device: {device}")
# ---------------------------------- #A10

# ---------------------------------- #A10
def param_transform(raw_params):
    """
    Map the original param_tensor (shape [11]) to the true model parameters:
      [mu_x, mu_y, phi_x, phi_y, gamma,
       omega_x, alpha_x, beta_x,
       omega_y, alpha_y, beta_y]

    Use differentiable functions (e.g., exp, softmax, tanh) to ensure:
    - omega_x, omega_y > 0
    - alpha, beta >= 0 and alpha+beta < c (c is set to 0.95 here to avoid being too close to 1)
    - |phi| < 0.99 (adjustable)
    """
    raw_params = raw_params.double()
    (raw_mu_x, raw_mu_y, raw_phi_x, raw_phi_y, raw_gamma,
     raw_omega_x, raw_alpha_x, raw_beta_x,
     raw_omega_y, raw_alpha_y, raw_beta_y) = torch.unbind(raw_params, dim=0)
    
    mu_x = raw_mu_x
    mu_y = raw_mu_y
    gamma = raw_gamma
    
    phi_x = 0.99 * torch.tanh(raw_phi_x)
    phi_y = 0.99 * torch.tanh(raw_phi_y)
    
    # 1) Map omega with exp to ensure >0, and add 1e-6 as a lower bound
    omega_x = torch.exp(raw_omega_x) + 1e-6
    omega_y = torch.exp(raw_omega_y) + 1e-6
    
    # 2) alpha, beta >=0 and alpha+beta< c
    #    Change c from 0.999 to 0.95 to avoid alpha+beta being too close to 1
    c = 0.99  
    # See how to modify the model without changing C to make calibration more accurate
    
    # Cannot change from 0.999 to 0.95 here, because the true value is actually 0.95, which would reveal the ground truth
    # c = 0.96
    # c = 0.95
    ab_x = torch.softmax(torch.stack([raw_alpha_x, raw_beta_x]), dim=0)
    alpha_x = c * ab_x[0]
    beta_x  = c * ab_x[1]
    
    ab_y = torch.softmax(torch.stack([raw_alpha_y, raw_beta_y]), dim=0)
    alpha_y = c * ab_y[0]
    beta_y  = c * ab_y[1]
    
    out_params = torch.stack([
        mu_x, mu_y, phi_x, phi_y, gamma,
        omega_x, alpha_x, beta_x,
        omega_y, alpha_y, beta_y
    ])
    return out_params
# ---------------------------------- #A10


# ---------------------------------- #A10
def negative_log_likelihood_torch(param_tensor, data_tensor):
    """
    Compute the negative log-likelihood of 2D AR(1)-GARCH(1,1) using PyTorch, can be accelerated on GPU.
    Use local variables to store eps and sigma2 at each time step to avoid large in-place tensors.
    """
    param_tensor = param_tensor.double()
    data_tensor = data_tensor.double()
    
    transformed_params = param_transform(param_tensor)
    mu_x, mu_y, phi_x, phi_y, gamma, \
    omega_x, alpha_x, beta_x, \
    omega_y, alpha_y, beta_y = torch.unbind(transformed_params, dim=0)
    
    N, T, _ = data_tensor.shape
    const_term = 0.5 * torch.log(2.0 * torch.tensor(np.pi, device=device, dtype=torch.double))
    
    X = data_tensor[:, :, 0]  # shape (N,T)
    Y = data_tensor[:, :, 1]  # shape (N,T)
    
    # t=0
    eps_x_tm1 = X[:, 0] - mu_x
    eps_y_tm1 = Y[:, 0] - mu_y
    sigma2_x_tm1 = omega_x / (1.0 - alpha_x - beta_x)
    sigma2_y_tm1 = omega_y / (1.0 - alpha_y - beta_y)
    
    neg_ll = torch.zeros((), device=device, dtype=torch.double)
    
    # Start calculation from t=1..T-1
    for t in range(1, T):
        eps_x_t = X[:, t] - (mu_x + phi_x * X[:, t-1])
        eps_y_t = Y[:, t] - (mu_y + phi_y * Y[:, t-1] + gamma * X[:, t-1])
        
        sigma2_x_t = omega_x + alpha_x*(eps_x_tm1**2) + beta_x*sigma2_x_tm1
        sigma2_y_t = omega_y + alpha_y*(eps_y_tm1**2) + beta_y*sigma2_y_tm1
        
        ll_x = const_term + 0.5*torch.log(sigma2_x_t) + 0.5*(eps_x_t**2 / sigma2_x_t)
        ll_y = const_term + 0.5*torch.log(sigma2_y_t) + 0.5*(eps_y_t**2 / sigma2_y_t)
        neg_ll += torch.sum(ll_x + ll_y)
        
        # Update previous time step
        eps_x_tm1 = eps_x_t
        eps_y_tm1 = eps_y_t
        sigma2_x_tm1 = sigma2_x_t
        sigma2_y_tm1 = sigma2_y_t
    
    # ---------------------------------- #A200
    # Without changing c=0.99, use a log barrier function
    # Encourage alpha+beta not to approach c, and do not need to use the true value 0.95 or threshold 0.98.
    penalty_weight = 1e-1   # Penalty coefficient, can be adjusted as needed
    epsilon = 1e-12         # Small offset to prevent log(0) numerical error

    sum_ab_x = alpha_x + beta_x  # alpha+beta for X channel
    sum_ab_y = alpha_y + beta_y  # alpha+beta for Y channel

    # As sum_ab_x approaches 0.99, (0.99 - sum_ab_x) becomes smaller, log(...) -> -∞, overall penalty -> +∞
    barrier_x = -torch.log((0.99 - sum_ab_x).clamp_min(epsilon))
    barrier_y = -torch.log((0.99 - sum_ab_y).clamp_min(epsilon))

    # Multiply by N: the more samples, the stronger the penalty, making optimization less likely to approach the boundary
    neg_ll += penalty_weight * (barrier_x + barrier_y) * N
    # ---------------------------------- #A200
    
    return neg_ll
# ---------------------------------- #A10

# ---------------------------------- #A10
def calibrate_2d_ar_garch_torch(data, max_iter=200):
    """
    Calibrate parameters using PyTorch's L-BFGS optimizer on GPU/CPU,
    - Change the upper bound of alpha+beta from 0.999 to 0.95 (see param_transform)
    - Lower the learning rate to 1e-3 for stable convergence
    - Set initial raw_omega_x, raw_omega_y to -2.0 => exp(-2)=0.135
    """
    data_tensor = torch.from_numpy(data).to(device)
    
    # init_params = np.array([
    #     0.0,  0.0,   # raw_mu_x, raw_mu_y
    #     0.0,  0.0,   # raw_phi_x, raw_phi_y
    #     0.0,         # raw_gamma
    #     -2.0,  0.0,  0.0,  # raw_omega_x, raw_alpha_x, raw_beta_x  => exp(-2)=0.135
    #     -2.0,  0.0,  0.0   # raw_omega_y, raw_alpha_y, raw_beta_y
    # ], dtype=np.float64)
    init_params = np.random.normal(loc=0.0, scale=1.0, size=11)
    # print("+++ INIT_PARAMS: ", init_params)
    
    # Convert initial parameters to true parameters for later display
    param_tensor_init = torch.tensor(init_params, device=device, dtype=torch.double)
    with torch.no_grad():
        init_transformed = param_transform(param_tensor_init)
        init_params_real = init_transformed.cpu().numpy()
    
    param_tensor = torch.tensor(init_params, device=device, requires_grad=True, dtype=torch.double)
    # Lower learning rate
    # optimizer = torch.optim.LBFGS([param_tensor], lr=1e-3, max_iter=max_iter, line_search_fn='strong_wolfe')
    optimizer = torch.optim.LBFGS(
        [param_tensor],
        lr=1e-3,
        max_iter=max_iter,
        line_search_fn='strong_wolfe',
        # ↓↓↓ The following thresholds are to be reduced ↓↓↓
        tolerance_grad=1e-9,     # Default is usually 1e-7, can be set smaller
        tolerance_change=1e-12,  # Default is usually 1e-9, can be set smaller
        history_size=10            # History size can be increased/decreased as appropriate
    )
    
    pbar = tqdm(total=max_iter, desc="LBFGS Optimization")
    iteration_count = [0]
    current_loss = [float('inf')]
    
    def closure():
        optimizer.zero_grad()
        loss = negative_log_likelihood_torch(param_tensor, data_tensor)
        loss.backward()
        
        iteration_count[0] += 1
        current_loss[0] = loss.item()
        pbar.set_postfix(loss=f"{current_loss[0]:.6f}")
        pbar.update(1)
        return loss
    
    try:
        final_loss = optimizer.step(closure)
        success = True
        message = "Optimization terminated successfully."
    except Exception as e:
        final_loss = torch.tensor(float('inf'), device=device, dtype=torch.double)
        success = False
        message = str(e)
    finally:
        pbar.close()
    
    raw_opt_params = param_tensor.detach().cpu().numpy()
    with torch.no_grad():
        final_transformed = param_transform(param_tensor)
        real_params_array = final_transformed.cpu().numpy()
    
    return {
        'raw': raw_opt_params,
        'x': real_params_array,
        'fun': float(final_loss.detach().cpu().numpy()),
        'success': success,
        'message': message,
        'iterations': iteration_count[0],
        'init_x': init_params_real  # Add the true value of the initial parameters
    }
# ---------------------------------- #A20


def get_training_data():
    print(">>> Step 1: Loading training data from diffwave module")
    data_loader, scalers, test_size = load_data()
    train_loader, _, _ = data_loader
    
    train_data_list = []
    for batch in train_loader:
        batch_data = batch[0].numpy()
        train_data_list.append(batch_data)
    train_data = np.concatenate(train_data_list, axis=0)
    
    # Save original scale data for reference and debugging
    train_data_inversed = inverse_transform_channels(train_data, scalers)
    
    print(f"Normalized training data shape: {train_data.shape}")
    print(f"Original scale training data shape: {train_data_inversed.shape}")
    
    # Return normalized data instead of original scale data, consistent with diffusion model
    return train_data, scalers, test_size, data_loader


def calibrate_and_save_model(train_data, max_samples=None):
    print(">>> Step 2: Calibrating GARCH model parameters")
    
    if max_samples is not None:
        calibration_sample_size = min(max_samples, len(train_data))
        calibration_subset = train_data[:calibration_sample_size]
        print(f"Using {calibration_sample_size} samples for calibration...")
    else:
        calibration_subset = train_data
        print(f"Using all training data ({len(train_data)} samples) for calibration...")
    
    print("Starting calibration...")
    result = calibrate_2d_ar_garch_torch(calibration_subset, max_iter=500)
    print("Calibration completed")
    
    param_names = [
        'mu_x', 'mu_y', 'phi_x', 'phi_y', 'gamma',
        'omega_x', 'alpha_x', 'beta_x',
        'omega_y', 'alpha_y', 'beta_y'
    ]
    
    print("\nCalibration Results:")
    print("=" * 40)
    for name, value in zip(param_names, result['x']):
        print(f"{name}: {value:.6f}")
    print(f"Final negative log-likelihood: {result['fun']:.6f}")
    print(f"Optimization success: {result['success']}")
    print(f"Optimization status: {result['message']}")
    print(f"Iterations: {result['iterations']}")
    print("=" * 40)
    
    # Add table comparing true values
    true_params = {
        'mu_x': 0.0, 'mu_y': 0.0,
        'phi_x': 0.95, 'phi_y': 0.90,
        'gamma': 0.8,
        'omega_x': 0.05, 'alpha_x': 0.05, 'beta_x': 0.90,
        'omega_y': 0.05, 'alpha_y': 0.05, 'beta_y': 0.90
    }
    
    print("\nComparison between calibrated and true values:")
    print("=" * 105)
    print(f"{'Parameter':<10}{'Initial Param':<15}{'Calibrated':<15}{'True Value':<15}{'Difference':<15}{'Percent Error':<15}")
    print("-" * 105)
    
    for i, name in enumerate(param_names):
        value = result['x'][i]
        init_value = result['init_x'][i]
        true_value = true_params[name]
        diff = value - true_value
        if true_value == 0:
            pct_err = "N/A"
        else:
            pct_err = f"{(diff/true_value)*100:.2f}%"
        
        print(f"{name:<10}{init_value:<15.6f}{value:<15.6f}{true_value:<15.6f}{diff:<15.6f}{pct_err:<15}")
    
    print("-" * 105)
    print("Notes:")
    print("1. AR parameters (phi_x, phi_y, gamma) are close to true values")
    print("2. GARCH parameters differ, but note the sum of alpha+beta:")
    print(f"   Calibrated: alpha_x + beta_x = {result['x'][6] + result['x'][7]:.4f}, alpha_y + beta_y = {result['x'][9] + result['x'][10]:.4f}")
    print(f"   True values: alpha_x + beta_x = {true_params['alpha_x'] + true_params['beta_x']:.4f}, alpha_y + beta_y = {true_params['alpha_y'] + true_params['beta_y']:.4f}")
    print("=" * 105)
    
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
    
    params_dict = dict(zip(param_names, result['x']))
    params_dict['timestamp'] = time_stamp
    params_dict['neg_log_likelihood'] = result['fun']
    
    save_dir = "calibrated_models"
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(params_dict, f"{save_dir}/garch_params_{time_stamp}.joblib")
    print(f"Model parameters saved to: {save_dir}/garch_params_{time_stamp}.joblib")
    
    return params_dict, time_stamp


def simulate_from_calibrated_model(params, T, N, scalers):
    print(f">>> Step 3: Simulating {N} samples with length {T} based on calibrated parameters")
    mu_x = params['mu_x']
    mu_y = params['mu_y']
    phi_x = params['phi_x']
    phi_y = params['phi_y']
    gamma = params['gamma']
    omega_x = params['omega_x']
    alpha_x = params['alpha_x']
    beta_x = params['beta_x']
    omega_y = params['omega_y']
    alpha_y = params['alpha_y']
    beta_y = params['beta_y']
    
    S_sim = np.zeros((N, T, 2))
    
    for n in tqdm(range(N), desc="Generating samples"):
        X = np.zeros(T)
        Y = np.zeros(T)
        
        epsilon_x = np.zeros(T)
        epsilon_y = np.zeros(T)
        sigma2_x = np.zeros(T)
        sigma2_y = np.zeros(T)
        
        z_x = np.random.randn(T)
        z_y = np.random.randn(T)
        
        sigma2_x[0] = omega_x / (1.0 - alpha_x - beta_x)
        sigma2_y[0] = omega_y / (1.0 - alpha_y - beta_y)
        
        epsilon_x[0] = np.sqrt(sigma2_x[0]) * z_x[0]
        epsilon_y[0] = np.sqrt(sigma2_y[0]) * z_y[0]
        
        X[0] = mu_x + epsilon_x[0]
        Y[0] = mu_y + epsilon_y[0]
        
        for t in range(1, T):
            sigma2_x[t] = omega_x + alpha_x*(epsilon_x[t-1]**2) + beta_x*sigma2_x[t-1]
            sigma2_y[t] = omega_y + alpha_y*(epsilon_y[t-1]**2) + beta_y*sigma2_y[t-1]
            
            epsilon_x[t] = np.sqrt(sigma2_x[t]) * z_x[t]
            epsilon_y[t] = np.sqrt(sigma2_y[t]) * z_y[t]
            
            X[t] = mu_x + phi_x*X[t-1] + epsilon_x[t]
            Y[t] = mu_y + phi_y*Y[t-1] + gamma*X[t-1] + epsilon_y[t]
        
        S_sim[n, :, 0] = X
        S_sim[n, :, 1] = Y
    
    # return S_sim
    
    # Simulated data in original scale
    simulated_data_original = S_sim.copy()
    
    # Apply normalization so that the data is on the same scale as the diffusion model output
    # 1. Separate the two channels
    channel_x = simulated_data_original[:, :, 0].reshape(-1, 1)  # (N*T, 1)
    channel_y = simulated_data_original[:, :, 1].reshape(-1, 1)  # (N*T, 1)
    
    # 2. Apply standardization transform to each channel
    channel_x_norm = scalers['x'].transform(channel_x)  # (N*T, 1)
    channel_y_norm = scalers['y'].transform(channel_y)  # (N*T, 1)
    
    # 3. Reshape back to (N, T) and stack to (N, T, 2)
    channel_x_norm = channel_x_norm.reshape(N, T)
    channel_y_norm = channel_y_norm.reshape(N, T)
    simulated_data_normalized = np.stack([channel_x_norm, channel_y_norm], axis=-1)
    
    # Return both scales of data
    return simulated_data_normalized, simulated_data_original


def analyze_simulated_data(simulated_data_normalized, simulated_data_original, params, time_stamp, scalers, data_loader):
    print(">>> Step 4: Analyzing simulated data")
    
    save_dir = "simulation_results"
    os.makedirs(save_dir, exist_ok=True)
    
    N, T, _ = simulated_data_normalized.shape
    
    # 1. Visualize several typical samples (using original scale data may be more intuitive)
    n_samples_to_plot = 5
    fig, axs = plt.subplots(n_samples_to_plot, 1, figsize=(12, 3*n_samples_to_plot))
    for i in range(n_samples_to_plot):
        axs[i].plot(simulated_data_original[i, :, 0], label='X Channel', color='blue')
        axs[i].plot(simulated_data_original[i, :, 1], label='Y Channel', color='red')
        axs[i].set_title(f'Sample {i+1} (Original Scale)')
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_paths_original_{time_stamp}.png")
    plt.close()
    
    # 2. Visualize samples in normalized scale
    fig, axs = plt.subplots(n_samples_to_plot, 1, figsize=(12, 3*n_samples_to_plot))
    for i in range(n_samples_to_plot):
        axs[i].plot(simulated_data_normalized[i, :, 0], label='X Channel', color='blue')
        axs[i].plot(simulated_data_normalized[i, :, 1], label='Y Channel', color='red')
        axs[i].set_title(f'Sample {i+1} (Normalized Scale)')
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_paths_normalized_{time_stamp}.png")
    plt.close()
    
    # 3. Save both scales of data
    np.save(f"{save_dir}/simulated_normalized_{time_stamp}.npy", simulated_data_normalized)
    np.save(f"{save_dir}/simulated_original_{time_stamp}.npy", simulated_data_original)
    
    # 4. Save scalers for later use
    joblib.dump(scalers, f"{save_dir}/scalers_{time_stamp}.joblib")
    
    print(f"Analysis results and simulated data saved to {save_dir} directory")
    print(f" - Normalized data (for comparison with diffusion models): {save_dir}/simulated_normalized_{time_stamp}.npy")
    print(f" - Original scale data: {save_dir}/simulated_original_{time_stamp}.npy")
    
    # 5. Try to compute evaluation metrics (if possible)
    try:
        from adaptive_ddpm import evaluate_metrics
        print("Computing evaluation metrics...")
        
        # Collect test data
        test_loader = data_loader[2]  # The third element of the data loader is the test set loader
        test_data_list = []
        for batch in test_loader:
            test_data_list.append(batch[0].numpy())
        test_data = np.concatenate(test_data_list, axis=0)
        print(f"Test data shape: {test_data.shape}")
        
        # Use normalized data to compute metrics, ensuring fair comparison with diffusion models
        if len(test_data) > len(simulated_data_normalized):
            test_data = test_data[:len(simulated_data_normalized)]
        elif len(test_data) < len(simulated_data_normalized):
            simulated_data_normalized = simulated_data_normalized[:len(test_data)]
            
        fro_diff, mse_val, mae_val, corr_val, wd_val = evaluate_metrics(test_data, simulated_data_normalized)
        print(f"Metrics (using normalized data):")
        print(f" - Frobenius Norm: {fro_diff:.4f}")
        print(f" - MSE: {mse_val:.4f}")
        print(f" - MAE: {mae_val:.4f}")
        print(f" - Correlation: {corr_val:.4f}")
        print(f" - Wasserstein Distance: {wd_val:.4f}")
        
        # Save evaluation results
        metrics = {
            'frobenius_norm': fro_diff,
            'mse': mse_val, 
            'mae': mae_val,
            'correlation': corr_val,
            'wasserstein': wd_val
        }
        metrics_file = f"{save_dir}/metrics_{time_stamp}.joblib"
        joblib.dump(metrics, metrics_file)
        print(f"Metrics saved to {metrics_file}")
    except (ImportError, AttributeError, NameError) as e:
        print(f"Evaluation metrics calculation failed: {e}")


def main():
    print("Starting GARCH model calibration and inference process...")
    
    # Step 1: Load normalized data and related information
    train_data, scalers, test_size, data_loader = get_training_data()
    print("Using normalized data for GARCH model calibration")
    
    # Step 2: Calibrate GARCH model parameters using normalized data
    params, time_stamp = calibrate_and_save_model(train_data, max_samples=80_000)
    print(f"GARCH model calibrated successfully with timestamp: {time_stamp}")
    
    # Step 3: Simulate using the calibrated model, return both normalized and original scale data
    T = train_data.shape[1]
    N = test_size  # Use the same number of samples as the test set for easy comparison
    print(f"Simulating {N} samples with time steps T={T}")
    simulated_data_normalized, simulated_data_original = simulate_from_calibrated_model(params, T, N, scalers)
    print("Simulation complete - generated both normalized and original scale data")
    
    # Step 4: Analyze and save results, including visualization and evaluation metrics
    analyze_simulated_data(simulated_data_normalized, simulated_data_original, params, time_stamp, scalers, data_loader)
    
    print("Execution completed! GARCH model has generated data in both normalized and original scales.")
    print("IMPORTANT: When comparing with diffusion models, always use normalized data for fair comparison.")


if __name__ == "__main__":
    # Enable for debugging if needed: torch.autograd.set_detect_anomaly(True)
    main()
