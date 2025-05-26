import numpy as np
from tqdm import tqdm

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

# Parameters for data generation
num_samples = 10000  # 10k samples
series_length = 100  # Each sample is of length 102

# Generate the dataset: each row is one time series sample
data = np.array([generate_ar1_garch_series(series_length) for _ in tqdm(range(num_samples))])

# Optional: Save the generated data to a CSV file
# np.savetxt("ar1_garch_data.csv", data, delimiter=",")
np.save("ar1_garch1_1_data_calibrated.npy", data)


