# -*- coding: utf-8 -*-

# Added: import necessary libraries
import os
import numpy as np
# Added: import iisignature
import iisignature as _iisig
# Added: import pandas for table output
import pandas as pd

# Added: helper function _to_numpy (from double_check_calibrated_garch.py)
def _to_numpy(arr) -> np.ndarray:
    """Ensure *arr* is a NumPy array on CPU."""
    if "torch" in str(type(arr)):
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
        except ImportError:
            pass
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Input must be a numpy array or torch tensor, got {type(arr)}")
    return arr

# Added: helper function compute_average_signature (from double_check_calibrated_garch.py)
def compute_average_signature(
    paths, level: int
) -> np.ndarray | None:
    """Average truncated signature of *paths* (N, T, d)."""
    if _iisig is None:
        return None
    
    x = _to_numpy(paths)
    if x.ndim != 3:
        raise ValueError(f"paths must have shape (N, T, d), but got {x.shape}")
    n_paths, _, dim_paths = x.shape
    
    try:
        sig_len = _iisig.siglength(int(dim_paths), int(level))
        all_signatures = _iisig.sig(x, level) 
        
        if all_signatures.ndim == 1 and n_paths == 1:
            all_signatures = all_signatures.reshape(1, sig_len)
        elif all_signatures.shape[0] != n_paths or all_signatures.shape[1] != sig_len:
            all_signatures = all_signatures.reshape(n_paths, sig_len)
            
        return all_signatures.mean(axis=0)
    except Exception as e:
        print(f"Error occurred while computing signature: {e}")
        return None

# Added: Path Signature Distance function
def path_signature_distance(
    paths_a,
    paths_b,
    level: int = 3,
) -> float | None:
    """
    Compute the average signature distance (Euclidean distance) between two sets of paths.
    (3) to measure distance between the stochastic processes we generate, 
        we might want to compute the signature (up to a given level; 
        for which there are packages that do it) and then compute the average 
        across elements and compute the Euclidean distance between these vectors.
    """
    if _iisig is None:
        print("iisignature module not loaded, cannot compute signature distance.")
        return np.nan # or None, depending on subsequent processing

    if paths_a.shape[2] != paths_b.shape[2]:
        raise ValueError(
            f"Path dimensions do not match: paths_a dimension {paths_a.shape[2]}, "
            f"paths_b dimension {paths_b.shape[2]}"
        )

    mean_sig_a = compute_average_signature(paths_a, level)
    mean_sig_b = compute_average_signature(paths_b, level)

    if mean_sig_a is None or mean_sig_b is None:
        return np.nan # or None

    # Compute Euclidean distance
    distance = np.linalg.norm(mean_sig_a - mean_sig_b)
    return float(distance)

# --- Main program: load data, compute and compare Signature distances --- #
def main():
    # Define real data and generated data information
    real_data_path = "test_samples_normalized_2025-03-28_05:11.npy"
    
    # Get hardcoded model filenames from ddpm_evaluate_and_plot.py
    model_filenames = [
        "best_model_adaptive_ddpm_20250511_191107.pth"
    ]
    
    # Define Signature Levels to compute
    signature_levels = [1, 2, 3, 4]
    
    # Define data dimensions to compare
    data_dimensions_to_compare = ['Channel 1', 'Channel 2', 'Combined Channels']
    
    if not model_filenames:
        print("No model filenames found for comparison.")
        return
    
    # Check if real data file exists
    if not os.path.exists(real_data_path):
        print(f"Error: Real data file not found: {real_data_path}")
        return
    
    # Load real data
    print(f"Loading real data: {real_data_path}")
    real_data_full = np.load(real_data_path)
    print(f"Shape of full real data: {real_data_full.shape}")
    
    # Store results of comparisons for different dimensions
    all_dimensions_results = {}
    
    for dim_case in data_dimensions_to_compare:
        print(f"\n--- Computing Signature distance for {dim_case} ---")
        
        current_dimension_results = []
        
        for model_fname in model_filenames:
            # Construct result directory name based on model filename
            dir_name = model_fname.replace("best_model_adaptive_", "results_").replace(".pth", "")
            generated_data_dir = os.path.join("./", dir_name) # Assume result directory is in current directory
            
            # print(f"Checking directory: {generated_data_dir}") # Reduce output
            
            if not os.path.isdir(generated_data_dir):
                # print(f"  Warning: Result directory not found: {generated_data_dir}. Skipping this model.") # Reduce output
                result_entry = {'Model Name': model_fname, 'Generated File': 'Directory not found'}
                for level in signature_levels:
                     result_entry[f'Signature Distance (Level {level})'] = np.nan
                current_dimension_results.append(result_entry)
                continue
                
            # Find .npy files in result directory
            npy_files_in_dir = [f for f in os.listdir(generated_data_dir) if f.endswith('.npy')]
            
            if not npy_files_in_dir:
                # print(f"  Warning: No .npy file found in directory {generated_data_dir}. Skipping this model.") # Reduce output
                result_entry = {'Model Name': model_fname, 'Generated File': 'No .npy file found'}
                for level in signature_levels:
                     result_entry[f'Signature Distance (Level {level})'] = np.nan
                current_dimension_results.append(result_entry)
                continue
                
            # Assume only one relevant .npy file per result directory, take the first found
            generated_fname = npy_files_in_dir[0]
            generated_file_path = os.path.join(generated_data_dir, generated_fname)
            
            # print(f"  Found generated .npy file: {generated_fname}") # Reduce output
            
            # Load generated data
            try:
                generated_data_full = np.load(generated_file_path)
                # print(f"  Shape of full generated data: {generated_data_full.shape}") # Reduce output
                
                # Extract data according to dim_case
                if dim_case == 'Channel 1':
                    real_data_slice = real_data_full[:, :, 0].reshape(real_data_full.shape[0], -1, 1)
                    generated_data_slice = generated_data_full[:, :, 0].reshape(generated_data_full.shape[0], -1, 1)
                    # print(f"    Extracted Channel 1 data shape: {real_data_slice.shape}, {generated_data_slice.shape}") # Reduce output
                elif dim_case == 'Channel 2':
                    real_data_slice = real_data_full[:, :, 1].reshape(real_data_full.shape[0], -1, 1)
                    generated_data_slice = generated_data_full[:, :, 1].reshape(generated_data_full.shape[0], -1, 1)
                    # print(f"    Extracted Channel 2 data shape: {real_data_slice.shape}, {generated_data_slice.shape}") # Reduce output
                elif dim_case == 'Combined Channels':
                    real_data_slice = real_data_full
                    generated_data_slice = generated_data_full
                    # print(f"    Using Combined Channels data shape: {real_data_slice.shape}, {generated_data_slice.shape}") # Reduce output
                else:
                     print(f"Unknown data dimension case: {dim_case}. Skipping.")
                     continue
                
                # Compute Signature distances for different Levels
                current_model_results = {'Model Name': model_fname, 'Generated File': generated_fname}
                any_success = False
                for level in signature_levels:
                    distance = path_signature_distance(real_data_slice, generated_data_slice, level=level)
                    if distance is not None and not np.isnan(distance):
                        # print(f"  Signature distance ({dim_case}, level {level}): {distance:.6f}") # Reduce output
                        current_model_results[f'Signature Distance (Level {level})'] = distance
                        any_success = True
                    else:
                        # print(f"  Signature distance ({dim_case}, level {level}): computation failed or not performed.") # Reduce output
                        current_model_results[f'Signature Distance (Level {level})'] = np.nan
                        
                # Add result even if all Levels failed, show NaN or error
                current_dimension_results.append(current_model_results)
                    
            except Exception as e:
                print(f"  Error occurred while processing file {generated_file_path}: {e}")
                error_result = {'Model Name': model_fname, 'Generated File': generated_fname}
                for level in signature_levels:
                     error_result[f'Signature Distance (Level {level})'] = f'Error: {e}'
                current_dimension_results.append(error_result)
                
        # Store current dimension's results in the overall results dictionary
        all_dimensions_results[dim_case] = pd.DataFrame(current_dimension_results)
        
    # Print all result tables
    print("\n" + "="*40)
    print("====== Final Signature Distance Comparison Results ======")
    print("="*40 + "\n")
    
    for dim_case, results_df in all_dimensions_results.items():
        print(f"====== {dim_case} Signature Distance ======")
        if not results_df.empty:
            # Can sort by Signature Distance (Level 3) if the column exists
            sort_column = 'Signature Distance (Level 3)'
            if sort_column in results_df.columns:
                 results_df = results_df.sort_values(by=sort_column, ascending=True)
            print(results_df.to_string(index=False))
        else:           
            print("\nNo Signature Distance results computed.")
        print("\n" + "-"*40 + "\n") # Separate different tables
        
    print("====== END OF ALL RESULTS ======")

# Run main function
if __name__ == "__main__":
    main()
