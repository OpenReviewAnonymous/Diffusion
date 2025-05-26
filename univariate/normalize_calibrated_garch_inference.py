import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime

# File list
input_files = [
    "calibrated_garch_20240815_164550.npy",
    "calibrated_garch_20240815_164613.npy"
]

# Same parameters as in diffwave.py
N = 100 * 1000
T = 100

def get_scaler_for_normalization():
    """
    Get the same scaler as load_data() in diffwave.py
    """
    # Load original data
    S_true = np.load("ar1_garch1_1_data.npy")
    
    # Same data processing as in diffwave.py
    # Calculate overall 99% quantile
    percentile_99 = np.percentile(S_true, 99)
    
    # Find which rows have values exceeding this quantile
    mask_exceed = np.any(np.abs(S_true) > percentile_99, axis=(1, 2))
    
    # Find rows with exceeding values
    indices_exceed = np.where(mask_exceed)[0]
    
    # Keep rows without exceeding values
    indices_clean = np.where(~mask_exceed)[0]
    S_clean = S_true[indices_clean]
    
    # Limit data size same as diffwave.py
    if len(S_clean) > N:
        S_clean = S_clean[:N]
    
    # Same data split as in diffwave.py
    train_ratio, val_ratio = 0.7, 0.15
    train_size = int(train_ratio * len(S_clean))
    val_size = int(val_ratio * len(S_clean))
    
    # Only use training data to fit scaler
    train_data = S_clean[:train_size]
    
    # Create same scaler as in diffwave.py
    scaler = StandardScaler()
    train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
    scaler.fit(train_data_reshaped)
    
    return scaler

"""
Use the same scaler as in diffwave.py to standardize all files
"""
# Get the same scaler as in diffwave.py
scaler = get_scaler_for_normalization()

# Create log file
log_filename = f"normalization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_filename, 'w', encoding='utf-8') as f:
    f.write(f"Normalization processing start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Scaler mean used: {scaler.mean_}\n")
    f.write(f"Scaler std used: {scaler.scale_}\n\n")
    
    # Apply standardization to each file and save
    for filename in tqdm(input_files, desc="Normalizing files"):
        # Load data
        data = np.load(filename)
        
        # Use scaler from diffwave.py for standardization
        data_reshaped = data.reshape(-1, data.shape[-1])
        normalized_data_reshaped = scaler.transform(data_reshaped)
        normalized_data = normalized_data_reshaped.reshape(data.shape)
        
        # Create output filename
        output_filename = filename.replace('.npy', '_normalized.npy')
        
        # Save standardized data
        np.save(output_filename, normalized_data)
        
        # Write to log
        f.write(f"Processed: {filename} -> {output_filename}\n")
        f.write(f"  Original data shape: {data.shape}\n")
        f.write(f"  Original data range: [{data.min():.4f}, {data.max():.4f}]\n")
        f.write(f"  Normalized range: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]\n\n")
    
    # Complete standardization
    f.write(f"Normalization processing completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"All files have been normalized. Check log file: {log_filename}")