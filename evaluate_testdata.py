import numpy as np
import torch
from diffwave import load_data, N, T  # Import necessary components from original file
from datetime import datetime

def save_test_samples():
    # Step 1: Get standardized test data through diffwave's data loading function
    data_loader_tuple, scaler, original_test_size = load_data()  # Get original data loader tuple
    
    # Step 2: Directly get original test tensor (skip DataLoader)
    test_dataset = data_loader_tuple[2].dataset  # Get test set in TensorDataset format
    test_tensor = test_dataset.tensors[0]  # Directly access complete tensor
    
    # Step 3: Convert to numpy array and extract required samples
    test_normalized = test_tensor.numpy()
    N_test = int(0.12 * N)
    
    # Automatically handle insufficient samples
    if len(test_normalized) < N_test:
        print(f"Warning: Available test samples ({len(test_normalized)}) < required ({N_test}), using all")
        N_test = len(test_normalized)
    
    # Step 4: Save standardized test data
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
    save_path = f"test_samples_normalized_{timestamp}.npy"
    np.save(save_path, test_normalized[:N_test])  # Take first N_test samples in order
    print(f"Saved {N_test} normalized test samples to {save_path}")

    # Step 5: Get and save original scale data
    # raw_test_data = scaler.inverse_transform(test_normalized[:N_test])
    # raw_save_path = f"test_samples_raw_{timestamp}.npy"
    # np.save(raw_save_path, raw_test_data)
    # print(f"Saved {N_test} raw test samples to {raw_save_path}")

if __name__ == "__main__":
    save_test_samples()