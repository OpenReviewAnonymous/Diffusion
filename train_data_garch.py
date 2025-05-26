import numpy as np
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from diffwave import load_data, inverse_transform_channels


###############################################################################
# 1) Parameter Initialization: AR(1)-GARCH(1,1)
###############################################################################
def initialize_garch_params_ar1(series):
    """
    Returns 5 parameters for AR(1)-GARCH(1,1):
    [Const, (AR(1)), omega, alpha, beta].
    We use a factor of 0.5 to get a bigger guess for omega, but you can tweak it.
    """
    try:
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        values = series.values.flatten()
        const_init = np.mean(values)
        
        # Rolling std for initial volatility guess
        rolling_std = series.rolling(20, min_periods=10).std().dropna()
        if len(rolling_std) > 0:
            omega_init = (np.median(rolling_std) ** 2) * 0.5
        else:
            omega_init = 0.5
        
        # The other parameters are guessed
        ar_init = 0.5
        alpha_init = 0.05 + np.random.rand() * 0.05
        beta_init  = 0.85 - np.random.rand() * 0.05
        
        return np.array([const_init, ar_init, max(omega_init, 1e-6), alpha_init, beta_init])
    except Exception as e:
        print(f"[WARN] Parameter init failed, using defaults: {e}")
        return np.array([0.0, 0.5, 0.1, 0.05, 0.85])


###############################################################################
# 2) Rolling Validation (No enforce_stationarity, No method=...)
###############################################################################
def rolling_validation_calibration(data, n_splits=5):
    """
    Perform rolling validation for AR(1)-GARCH(1,1).
    We do NOT pass enforce_stationarity or method=... to arch_model/fit,
    so it will work on older arch versions.
    """
    params_list = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Flatten data to 1D
    if isinstance(data, (pd.DataFrame, np.ndarray)):
        data = data.reshape(-1)
    elif isinstance(data, pd.Series):
        data = data.values.reshape(-1)
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(data)):
        try:
            train_data = data[train_idx].squeeze()
            
            init_params = initialize_garch_params_ar1(train_data)
            model = arch_model(
                train_data,
                mean='AR',
                lags=1,
                vol='GARCH',
                p=1, q=1,
                dist='normal'
            )
            
            res = model.fit(
                update_freq=10,
                disp='off',
                show_warning=True,
                starting_values=init_params,
                options={'maxiter': 500}
            )
            
            print(f"[Iteration {i+1}] param index:", res.params.index)
            
            fitted_params = res.params.copy()
            param_index = set(fitted_params.index)
            required_set = {'Const', 'omega', 'alpha[1]', 'beta[1]'}
            
            # Check if all required volatility params exist
            missing_req = required_set - param_index
            if missing_req:
                raise ValueError(f"Missing required param(s): {missing_req}")
            
            # The leftover param is the AR(1) coefficient
            leftover = param_index - required_set
            if len(leftover) != 1:
                raise ValueError(f"Could not identify AR(1) param uniquely. leftover={leftover}")
            
            ar_name = leftover.pop()
            # rename to 'ar.L1'
            fitted_params['ar.L1'] = fitted_params[ar_name]
            if ar_name != 'ar.L1':
                fitted_params.drop(ar_name, inplace=True)
            
            params_list.append(fitted_params)
        except Exception as e:
            print(f"[Iteration {i+1}] rolling validation failed: {e}")
            continue
    
    if not params_list:
        raise RuntimeError("All rolling validation iterations failed.")
    
    # Average across successful fits
    return pd.DataFrame(params_list).mean().to_dict()


###############################################################################
# 3) 2D GARCH Calibration: X=AR(1)-GARCH(1,1), Y=ARX(1)-GARCH(1,1)
###############################################################################
def calibrate_2d_ar_garch(calibration_subset):
    try:
        # Flatten X and Y
        X_samples = calibration_subset[:, :, 0].reshape(-1)
        Y_samples = calibration_subset[:, :, 1].reshape(-1)
        
        # Calibrate X
        params_x = rolling_validation_calibration(X_samples)
        print(f"[INFO] X model average params: {params_x}")
        
        # Calibrate Y (ARX(1)-GARCH(1,1))
        exog_df = pd.DataFrame({'X_lag1': X_samples[:-1]})
        model_y = arch_model(
            Y_samples[1:], 
            x=exog_df,
            mean='ARX',
            lags=1,
            vol='GARCH',
            p=1, q=1,
            dist='normal'
        )
        
        init_params_y = initialize_garch_params_ar1(Y_samples)
        # add gamma guess (the cross effect)
        init_params_y = np.append(init_params_y, 0.5)
        
        res_y = model_y.fit(
            update_freq=10,
            disp='off',
            show_warning=True,
            starting_values=init_params_y,
            options={'maxiter': 500}
        )
        
        fitted_params_y = res_y.params.copy()
        print("[INFO] Y param index:", fitted_params_y.index)
        
        # We expect 'Const','X_lag1','omega','alpha[1]','beta[1]' plus leftover for AR(1)
        base_required = {'Const', 'X_lag1', 'omega', 'alpha[1]', 'beta[1]'}
        param_index_y = set(fitted_params_y.index)
        
        missing_req_y = base_required - param_index_y
        if missing_req_y:
            raise ValueError(f"Missing required param(s) for Y: {missing_req_y}")
        
        leftover_y = param_index_y - base_required
        if len(leftover_y) != 1:
            raise ValueError(f"Could not identify AR(1) param for Y. leftover={leftover_y}")
        
        y_ar_name = leftover_y.pop()
        fitted_params_y['y.L1'] = fitted_params_y[y_ar_name]
        if y_ar_name != 'y.L1':
            fitted_params_y.drop(y_ar_name, inplace=True)
        
        print("[INFO] Y final params:\n", fitted_params_y)
        
        return {
            'mu_x':    params_x.get('Const', 0.0),
            'phi_x':   params_x.get('ar.L1', 0.0),
            'omega_x': params_x.get('omega', 0.0),
            'alpha_x': params_x.get('alpha[1]', 0.0),
            'beta_x':  params_x.get('beta[1]',  0.0),
            
            'mu_y':    fitted_params_y.get('Const', 0.0),
            'phi_y':   fitted_params_y.get('y.L1', 0.0),
            'gamma':   fitted_params_y.get('X_lag1', 0.5),
            'omega_y': fitted_params_y.get('omega', 0.0),
            'alpha_y': fitted_params_y.get('alpha[1]', 0.0),
            'beta_y':  fitted_params_y.get('beta[1]',  0.0)
        }
    except Exception as e:
        print(f"[ERROR] calibrate_2d_ar_garch failed: {e}")
        raise


###############################################################################
# 4) Load Training Data
###############################################################################
def get_training_data():
    print("[INFO] Loading training data...")
    try:
        data_loader, scalers, test_size = load_data()
        train_loader, _, _ = data_loader
        
        train_data_list = []
        for batch in train_loader:
            batch_data = batch[0].numpy()
            train_data_list.append(batch_data)
        
        train_data = np.concatenate(train_data_list, axis=0)
        train_data_inversed = inverse_transform_channels(train_data, scalers)
        print("[INFO] Train data shape:", train_data_inversed.shape)
        return train_data_inversed, scalers, test_size
    except Exception as e:
        print(f"[ERROR] get_training_data failed: {e}")
        raise


###############################################################################
# 5) Calibrate and Save
###############################################################################
def calibrate_and_save_model(train_data, max_samples=None):
    print("[INFO] Starting GARCH calibration (very old arch style)...")
    try:
        if max_samples is not None:
            calibration_sample_size = min(max_samples, len(train_data))
            calibration_subset = train_data[:calibration_sample_size]
        else:
            calibration_subset = train_data
        
        # ---------------------------------- #G1
        # (AR(1)-GARCH(1,1) Constraints):
        # (1) mu_x, mu_y in R
        # (2) |phi_x|<1, |phi_y|<1
        # (3) gamma in R (no strict bound)
        # (4) omega_x, omega_y > 0
        # (5) alpha_x, alpha_y >= 0
        # (6) beta_x, beta_y >= 0
        # (7) alpha_x + beta_x < 1, alpha_y + beta_y < 1
        #
        # These constraints must be satisfied during data generation or fitting:
        # - Ensure AR process is stationary (|phi|<1)
        # - Ensure GARCH process is stationary (omega>0, alpha+beta<1, alpha,beta>=0)
        # ---------------------------------- #G1
        
        result = calibrate_2d_ar_garch(calibration_subset)
        
        param_names = [
            'mu_x', 'phi_x', 'omega_x', 'alpha_x', 'beta_x',
            'mu_y', 'phi_y', 'gamma', 'omega_y', 'alpha_y', 'beta_y'
        ]
        
        # Example "true" reference
        true_params = {
            'mu_x': 0.0, 'phi_x': 0.95, 'omega_x': 0.05, 
            'alpha_x': 0.05, 'beta_x': 0.90,
            'mu_y': 0.0, 'phi_y': 0.90, 'gamma': 0.8,
            'omega_y': 0.05, 'alpha_y': 0.05, 'beta_y': 0.90
        }
        
        print("\n[REPORT] Parameter Comparison:")
        print("=" * 95)
        for name in param_names:
            est = result[name]
            truev = true_params[name]
            diff = est - truev
            if abs(truev) > 1e-12:
                pct_err = f"{abs(diff / truev)*100:.2f}%"
            else:
                pct_err = "N/A"
            print(f"{name:<10}{est:<14.6f}{truev:<14.6f}{diff:<14.6f}{pct_err:<14}")
        
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        params_dict = {k: result[k] for k in param_names}
        params_dict.update({
            'timestamp': time_stamp,
            'stability_x': result['alpha_x'] + result['beta_x'],
            'stability_y': result['alpha_y'] + result['beta_y']
        })
        
        save_dir = "calibrated_models"
        os.makedirs(save_dir, exist_ok=True)
        out_file = f"{save_dir}/garch_params_{time_stamp}.joblib"
        joblib.dump(params_dict, out_file)
        
        return params_dict, time_stamp
    except Exception as e:
        print(f"[ERROR] calibrate_and_save_model failed: {e}")
        raise


###############################################################################
# 6) Simulation
###############################################################################
def simulate_from_calibrated_model(params, T, N):
    print("[INFO] Starting simulation from calibrated model...")
    try:
        S_sim = np.zeros((N, T, 2))
        
        for n in tqdm(range(N), desc="Simulating"):
            X = np.zeros(T)
            Y = np.zeros(T)
            sigma2_x = np.zeros(T)
            sigma2_y = np.zeros(T)
            
            z_x = np.random.randn(T)
            z_y = np.random.randn(T)
            
            # Basic stationarity assumption: alpha+beta < 1
            denom_x = 1 - (params['alpha_x'] + params['beta_x'])
            denom_y = 1 - (params['alpha_y'] + params['beta_y'])
            
            sigma2_x[0] = params['omega_x'] / denom_x if denom_x != 0 else params['omega_x']
            sigma2_y[0] = params['omega_y'] / denom_y if denom_y != 0 else params['omega_y']
            
            X[0] = params['mu_x'] + np.sqrt(max(sigma2_x[0], 1e-12)) * z_x[0]
            Y[0] = params['mu_y'] + np.sqrt(max(sigma2_y[0], 1e-12)) * z_y[0]
            
            for t in range(1, T):
                sigma2_x[t] = (
                    params['omega_x']
                    + params['alpha_x'] * (X[t-1] - params['mu_x'])**2
                    + params['beta_x'] * sigma2_x[t-1]
                )
                sigma2_y[t] = (
                    params['omega_y']
                    + params['alpha_y'] * (Y[t-1] - params['mu_y'])**2
                    + params['beta_y'] * sigma2_y[t-1]
                )
                
                X[t] = (
                    params['mu_x']
                    + params['phi_x']*(X[t-1] - params['mu_x'])
                    + np.sqrt(max(sigma2_x[t], 1e-12))*z_x[t]
                )
                Y[t] = (
                    params['mu_y']
                    + params['phi_y']*(Y[t-1] - params['mu_y'])
                    + params['gamma']*(X[t-1] - params['mu_x'])
                    + np.sqrt(max(sigma2_y[t], 1e-12))*z_y[t]
                )
            
            S_sim[n, :, 0] = X
            S_sim[n, :, 1] = Y
        
        return S_sim
    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        raise


###############################################################################
# 7) Main
###############################################################################
def main():
    print("[MAIN] GARCH calibration process (compatible with older arch) started...")
    train_data, scalers, test_size = get_training_data()
    
    # Here you can choose max_samples=..., for example 80_000 / 8_000 / 800
    params, timestamp = calibrate_and_save_model(train_data, max_samples=80_000)
    
    T = train_data.shape[1]
    N = test_size
    simulated_data = simulate_from_calibrated_model(params, T, N)
    
    outdir = "simulation_results"
    os.makedirs(outdir, exist_ok=True)
    out_file = f"{outdir}/simulated_{timestamp}_size{N}.npy"
    np.save(out_file, simulated_data)
    print(f"[INFO] Simulated data saved to {out_file}")


if __name__ == "__main__":
    main()
