#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics calculation and visualization for time series generative models
"""

# =============== Import necessary libraries ===============
from networkx import difference
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, entropy
from sklearn.neighbors import KernelDensity
import ot  # Required: pip install pot
from sympy import Ordinal
from tqdm import tqdm
from datetime import datetime
from PIL import Image

import adaptive_ddpm

# New: import iisignature
import iisignature as _iisig


timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")


# Import from diffwave
try:
    from diffwave import N
    from diffwave import load_data
    N_test = int(0.12 * N)
except ImportError:
    print("Warning: Unable to import diffwave module")

# =============== Data loading functions ===============
def load_datasets():
    """Load all experimental datasets"""
    # Real data
    real_data = np.load("test_samples_normalized_2025-05-15_02:07.npy") # univariate
    print("Real data shape:", real_data.shape)
    
    
    # Try to load DDPM model generated data
    ddpm_data_path = "samples_ddpm_2025-04-23_17-43.npy"
    ddpm_data = np.load(ddpm_data_path)
    ddpm_model_name = ddpm_data_path.split("_")[1]  # Extract "ddpm"
    print(f"{ddpm_model_name} data shape:", ddpm_data.shape)

    
    # DiffWave model generated data
    diffwave_data_path = "samples_diffwave_2025-03-28_04:59.npy"
    diffwave_data = np.load(diffwave_data_path)
    diffwave_data =  diffwave_data
    diffwave_model_name = diffwave_data_path.split("_")[1]  # Extract "diffwave"
    print(f"{diffwave_model_name} data shape:", diffwave_data.shape)
    
    # Adaptive model generated data
    adaptive_data_path = "samples_adaptive_ddpm_2025-03-28_09-02.npy"
    adaptive_diffwave_data = np.load(adaptive_data_path)
    # Extract model name (adaptive_ddpm) from filename
    model_name_parts = adaptive_data_path.split("_")
    adaptive_model_name = f"{model_name_parts[1]}_{model_name_parts[2]}"  # Extract "adaptive_ddpm"
    print(f"{adaptive_model_name} data shape:", adaptive_diffwave_data.shape)
    
    # Load simulated data calibrated from file
    calibrated_data_path = "simulation_results/simulated_normalized_2025-05-11_06:17.npy"
    calibrated_data = np.load(calibrated_data_path)
    print(f"calibrated_data data shape:", calibrated_data.shape)
    
    # Ensure all data are in correct shape, for univariate data, if 2D (N,T), expand to 3D (N,T,1)
    real_data = _ensure_3d(real_data)
    diffwave_data = _ensure_3d(diffwave_data)
    adaptive_diffwave_data = _ensure_3d(adaptive_diffwave_data)
    calibrated_data = _ensure_3d(calibrated_data)
    ddpm_data = _ensure_3d(ddpm_data)
    
    return real_data, diffwave_data, adaptive_diffwave_data, diffwave_model_name, adaptive_model_name, calibrated_data, ddpm_data, ddpm_model_name

def _ensure_3d(data):
    """Ensure data is in 3D form (N,T,1), suitable for univariate data"""
    if data.ndim == 2:  # If shape is (N,T)
        return data.reshape(data.shape[0], data.shape[1], 1)
    elif data.ndim == 3 and data.shape[2] > 1:  # If shape is (N,T,D) and D>1, take only the first dimension
        return data[:, :, 0:1]
    return data

# =============== Data statistics and visualization functions ===============
def plot_correlation_matrix(data, title="Correlation Matrix", save_path=None, vmin=-1, vmax=1):
    """
    Plot the correlation matrix heatmap for 2D time series data
    
    Args:
    data: shape (N, T, 2), where N is number of samples, T is time steps, 2 is dimension
    title: chart title
    save_path: path to save the chart, if None use title as filename
    vmin, vmax: color range for heatmap
    """
    # Extract X and Y sequences from (N, T, 2)
    N, T, dim = data.shape
    X_flat = data[:, :, 0]  # Extract first dimension data, shape (N, T)
    Y_flat = data[:, :, 1]  # Extract second dimension data, shape (N, T)
    
    # Concatenate X and Y to shape (N, 2T)
    flatten_data = np.hstack([X_flat, Y_flat])
    
    # Compute correlation matrix, rowvar=False means each column is a variable
    corr_matrix = np.corrcoef(flatten_data, rowvar=False)
    
    # Plot correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap="RdBu_r",  # Red-blue contrast
        center=0,       # Center color at 0
        vmax=vmax, vmin=vmin,  # Use provided unified range
        square=True,
        fmt='.2f',
        cbar=True
    )
    
    # Draw dividing lines at T to split matrix into four blocks
    plt.axvline(x=T, color='k', linestyle='--')  # Vertical split
    plt.axhline(y=T, color='k', linestyle='--')  # Horizontal split
    
    # Add block labels
    plt.text(T/2, -5, "X-X", ha='center')
    plt.text(T+T/2, -5, "X-Y", ha='center')
    plt.text(-5, T/2, "Y-X", va='center', rotation=90)
    plt.text(-5, T+T/2, "Y-Y", va='center', rotation=90)
    
    plt.title(title)
    plt.tight_layout()
    
    # Save chart
    if save_path is None:
        save_path = f"{title.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation matrix plot saved to: {save_path}")
    return corr_matrix

def plot_covariance_matrix(data, title="Covariance Matrix", save_path=None, vmin=None, vmax=None):
    """
    Plot the covariance matrix heatmap for 2D time series data
    
    Args:
    data: shape (N, T, 2), where N is number of samples, T is time steps, 2 is dimension
    title: chart title
    save_path: path to save the chart, if None use title as filename
    vmin, vmax: color range for heatmap
    """
    # Extract X and Y sequences from (N, T, 2)
    N, T, dim = data.shape
    X_flat = data[:, :, 0]  # Extract first dimension data, shape (N, T)
    Y_flat = data[:, :, 1]  # Extract second dimension data, shape (N, T)
    
    # Concatenate X and Y to shape (N, 2T)
    flatten_data = np.hstack([X_flat, Y_flat])
    
    # Compute covariance matrix
    cov_matrix = np.cov(flatten_data, rowvar=False)
    
    # Plot covariance matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cov_matrix,
        cmap="RdBu_r",  # Red-blue contrast
        center=0,       # Center color at 0
        vmin=vmin, vmax=vmax,  # Use provided unified range
        square=True,
        fmt='.2f',
        cbar=True
    )
    
    # Draw dividing lines at T to split matrix into four blocks
    plt.axvline(x=T, color='k', linestyle='--')  # Vertical split
    plt.axhline(y=T, color='k', linestyle='--')  # Horizontal split
    
    # Add block labels
    plt.text(T/2, -5, "X-X", ha='center')
    plt.text(T+T/2, -5, "X-Y", ha='center')
    plt.text(-5, T/2, "Y-X", va='center', rotation=90)
    plt.text(-5, T+T/2, "Y-Y", va='center', rotation=90)
    
    plt.title(title)
    plt.tight_layout()
    
    # Save chart
    if save_path is None:
        save_path = f"{title.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Covariance matrix plot saved to: {save_path}")
    return cov_matrix

def plot_statistical_distributions(data, title="Statistical Distributions", save_path=None, xlims=None, ylims=None):
    """
    Plot statistical distributions for 2D time series data
    
    Args:
    data: shape (N, T, 2), where N is number of samples, T is time steps, 2 is dimension
    title: chart title
    save_path: path to save the chart, if None use title as filename
    xlims: list of x-axis ranges for each subplot, format [(xmin1, xmax1), (xmin2, xmax2), ...]
    ylims: list of y-axis ranges for each subplot, format [(ymin1, ymax1), (ymin2, ymax2), ...]
    """
    N, T, dim = data.shape
    
    # Compute statistics
    means_x = np.mean(data[:, :, 0], axis=1)  # Mean of X dimension
    means_y = np.mean(data[:, :, 1], axis=1)  # Mean of Y dimension
    
    vars_x = np.var(data[:, :, 0], axis=1)    # Variance of X dimension
    vars_y = np.var(data[:, :, 1], axis=1)    # Variance of Y dimension
    
    skew_x = skew(data[:, :, 0], axis=1)      # Skewness of X dimension
    skew_y = skew(data[:, :, 1], axis=1)      # Skewness of Y dimension
    
    # Create a 3x2 chart layout (3 statistics x 2 dimensions)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Mean distribution
    axes[0, 0].hist(means_x, bins=50, color='blue', alpha=0.7)
    axes[0, 0].set_title('X Dimension Mean Distribution')
    axes[0, 0].set_xlabel('Mean')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(means_y, bins=50, color='red', alpha=0.7)
    axes[0, 1].set_title('Y Dimension Mean Distribution')
    axes[0, 1].set_xlabel('Mean')
    axes[0, 1].set_ylabel('Frequency')
    
    # Variance distribution
    axes[1, 0].hist(vars_x, bins=50, color='blue', alpha=0.7)
    axes[1, 0].set_xlim(0, 10)
    axes[1, 0].set_title('X Dimension Variance Distribution')
    axes[1, 0].set_xlabel('Variance')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(vars_y, bins=50, color='red', alpha=0.7)
    axes[1, 1].set_xlim(0, 10)
    axes[1, 1].set_title('Y Dimension Variance Distribution')
    axes[1, 1].set_xlabel('Variance')
    axes[1, 1].set_ylabel('Frequency')
    
    # Skewness distribution
    axes[2, 0].hist(skew_x, bins=50, color='blue', alpha=0.7)
    axes[2, 0].set_title('X Dimension Skewness Distribution')
    axes[2, 0].set_xlabel('Skewness')
    axes[2, 0].set_ylabel('Frequency')
    
    axes[2, 1].hist(skew_y, bins=50, color='red', alpha=0.7)
    axes[2, 1].set_title('Y Dimension Skewness Distribution')
    axes[2, 1].set_xlabel('Skewness')
    axes[2, 1].set_ylabel('Frequency')
    
    # Set unified axis ranges
    if xlims is not None:
        for i in range(3):
            for j in range(2):
                idx = i * 2 + j
                if idx < len(xlims) and xlims[idx] is not None:
                    axes[i, j].set_xlim(xlims[idx])
                    
                    
    axes[1, 0].set_xlim(0, 10)
    axes[1, 1].set_xlim(0, 10)
    
    
    if ylims is not None:
        for i in range(3):
            for j in range(2):
                idx = i * 2 + j
                if idx < len(ylims) and ylims[idx] is not None:
                    axes[i, j].set_ylim(ylims[idx])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Leave space for the main title
    
    # Save chart
    if save_path is None:
        save_path = f"{title.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Statistical distribution plot saved to: {save_path}")
    return (means_x, means_y), (vars_x, vars_y), (skew_x, skew_y)

def data_summary(data, name="Dataset", save_path=None, save_dir="evaluation_figures", xlims=None, ylims=None):
    """Print basic statistics of the dataset, considering univariate data, and generate visualization chart"""
    print(f"\n====== {name} Statistics ======")
    print(f"Shape: {data.shape}")
    
    # Ensure data is 3D
    data = _ensure_3d(data)
    
    # Flatten univariate data
    flat_data = data[:, :, 0].flatten()
    
    # Statistics
    print(f"Statistics:")
    print(f"  Min: {flat_data.min():.4f}")
    print(f"  1% quantile: {np.quantile(flat_data, 0.01):.4f}")
    print(f"  25% quantile: {np.quantile(flat_data, 0.25):.4f}")
    print(f"  50% quantile: {np.median(flat_data):.4f}")
    print(f"  75% quantile: {np.quantile(flat_data, 0.75):.4f}")
    print(f"  99% quantile: {np.quantile(flat_data, 0.99):.4f}")
    print(f"  Max: {flat_data.max():.4f}")
    print(f"  Mean: {np.mean(flat_data):.4f}")
    print(f"  Std: {np.std(flat_data):.4f}")
    print(f"  Skewness: {skew(flat_data):.4f}")
    
    # Create visualization chart
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram
    axs[0].hist(flat_data, bins=50, color='blue', alpha=0.7)
    axs[0].set_title('Distribution Histogram')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')
    
    # Box plot
    axs[1].boxplot(flat_data, vert=False, widths=0.7)
    axs[1].set_title('Box Plot')
    axs[1].set_xlabel('Value')
    
    # Set unified axis ranges
    if xlims is not None:
        for i in range(2):
            if i < len(xlims) and xlims[i] is not None:
                axs[i].set_xlim(xlims[i])
    
    if ylims is not None:
        for i in range(2):
            if i < len(ylims) and ylims[i] is not None:
                axs[i].set_ylim(ylims[i])
    
    plt.suptitle(f"{name} Statistical Distribution", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the title
    
    # Save chart - ensure no spaces in filename
    file_name = name.replace(' ', '_')
    if save_path is None:
        save_path = f"{save_dir}/{file_name}_summary.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Data statistics plot saved to: {save_path}")

    return save_path

# Image merging function
def merge_images(image_paths, output_path, border_size=5, border_color=(255, 0, 0)):
    """
    Horizontally merge multiple images and add borders
    
    Args:
    image_paths: list of image paths
    output_path: output image path
    border_size: border width
    border_color: border color, RGB tuple
    """
    # Open all images
    images = [Image.open(p) for p in image_paths]
    
    # Get size of the first image
    width, height = images[0].size
    
    # Set layout: 1 row, n columns
    rows = 1
    cols = len(images)
    
    # Add border to images
    def add_border(image, border_size=5, color=(255, 0, 0)):
        """Add specified color and width border to a single image"""
        w, h = image.size
        # Create a new image slightly larger than the original to hold the border
        bordered_img = Image.new("RGB", (w + 2*border_size, h + 2*border_size), color)
        # Paste the original image in the center of the new image
        bordered_img.paste(image, (border_size, border_size))
        return bordered_img
    
    # Add border
    images_with_border = [add_border(img, border_size=border_size, color=border_color) for img in images]
    
    # Get size of images after adding border
    bordered_width, bordered_height = images_with_border[0].size
    
    # Create canvas for the final merged image
    combined_width = cols * bordered_width
    combined_height = rows * bordered_height
    merged_image = Image.new('RGB', (combined_width, combined_height))
    
    # Paste each sub-image in order into the large image
    for idx, img in enumerate(images_with_border):
        # Calculate which column the current image should be in
        col = idx % cols
        
        # Calculate the top-left pixel coordinates for pasting
        x_offset = col * bordered_width
        y_offset = 0  # Only one row, so y offset is 0
        
        # Paste the image at the specified position
        merged_image.paste(img, (x_offset, y_offset))
    
    # Save the final merged image
    merged_image.save(output_path)
    print(f"Merging complete, saved as {output_path}")
    return output_path

# =============== Evaluation metric calculation functions ===============

# New: helper function _to_numpy (from double_check_calibrated_garch.py)
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

# New: helper function compute_average_signature (from double_check_calibrated_garch.py)
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
        print(f"Error computing signature: {e}")
        return None

# New: Path Signature Distance function
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
        return np.nan # or None, depending on later handling

    # Ensure both sets of data are 3D
    paths_a = _ensure_3d(paths_a)
    paths_b = _ensure_3d(paths_b)

    if paths_a.shape[2] != paths_b.shape[2]:
        raise ValueError(
            f"Path dimension mismatch: paths_a dim {paths_a.shape[2]}, "
            f"paths_b dim {paths_b.shape[2]}"
        )

    mean_sig_a = compute_average_signature(paths_a, level)
    mean_sig_b = compute_average_signature(paths_b, level)

    if mean_sig_a is None or mean_sig_b is None:
        return np.nan # or None

    # Compute Euclidean distance
    distance = np.linalg.norm(mean_sig_a - mean_sig_b)
    return float(distance)

def correlation_matrix_difference(data1, data2):
    """
    Compute the Frobenius norm of the difference between two 2D data correlation matrices
    
    Args:
    data1, data2: datasets of shape (N, T, 2) and (M, T, 2)
    
    Returns:
    float: Frobenius norm of the correlation matrix difference
    """
    # Process data1
    N1, T, _ = data1.shape
    X1_flat = data1[:, :, 0]
    Y1_flat = data1[:, :, 1]
    flatten_data1 = np.hstack([X1_flat, Y1_flat])
    corr1 = np.corrcoef(flatten_data1, rowvar=False)
    
    # Process data2
    N2, T, _ = data2.shape
    X2_flat = data2[:, :, 0]
    Y2_flat = data2[:, :, 1]
    flatten_data2 = np.hstack([X2_flat, Y2_flat])
    corr2 = np.corrcoef(flatten_data2, rowvar=False)
    
    # Compute Frobenius norm of the difference
    diff = corr1 - corr2
    return np.linalg.norm(diff, ord='fro')  # Frobenius norm

def marginal_wasserstein_distance(real_data, gen_data, normalize=False):
    """
    Compute marginal Wasserstein distance (MarginalWD)
    
    MarginalWD computes the Wasserstein distance for each dimension separately, then averages,
    treating each dimension as an independent marginal distribution, ignoring inter-dimensional correlation.
    
    Args:
    real_data: real data of shape (N, T, D)
    gen_data: generated data of shape (M, T, D)
    normalize: whether to normalize by the standard deviation of real data
    
    Returns:
    float: marginal Wasserstein distance
    """
    N, T, D = real_data.shape
    
    total_wd = 0
    # Compute Wasserstein distance for each dimension separately
    for d in range(D):
        # Flatten each dimension's data to 1D
        real_dim = real_data[:, :, d].flatten()
        gen_dim = gen_data[:, :, d].flatten()
        
        # Normalize if needed
        if normalize:
            real_std = np.std(real_dim)
            if real_std > 0:  # Avoid division by zero
                real_dim = real_dim / real_std
                gen_dim = gen_dim / real_std
        
        # Use 1D Wasserstein distance (based on sorting, O(n log n) complexity)
        # This is much more efficient than general EMD, suitable for 1D data
        wd = ot.wasserstein_1d(real_dim, gen_dim)
        total_wd += wd
    
    # Return average Wasserstein distance across all dimensions
    return total_wd / D

def sliced_wasserstein_2d(X, Y, num_projections=5000, normalize=False, seed=0):
    """
    Compute 2D Sliced Wasserstein distance via Monte Carlo random direction projections
    Complexity O(K·(N log N + M log M)), K=num_projections

    Args:
      X, Y: np.ndarray, shape=(N,2)/(M,2)
      num_projections: number of projection directions
      normalize: whether to divide X,Y by std in each dimension
      seed: random seed
    Returns:
      float: average sliced Wasserstein distance
    """
    if normalize:
        stds = np.std(X, axis=0)
        nz = stds > 0
        X[:, nz] /= stds[nz]
        Y[:, nz] /= stds[nz]

    rng = np.random.RandomState(seed)
    angles = rng.rand(num_projections) * 2 * np.pi
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (K,2)

    PX = X.dot(dirs.T)  # (N, K)
    PY = Y.dot(dirs.T)  # (M, K)

    sw_total = 0.0
    for k in range(num_projections):
        a = np.sort(PX[:, k])
        b = np.sort(PY[:, k])
        # Interpolate to align if lengths differ
        if a.size > b.size:
            b = np.interp(np.linspace(0, 1, a.size),
                          np.linspace(0, 1, b.size), b)
        elif a.size < b.size:
            a = np.interp(np.linspace(0, 1, b.size),
                          np.linspace(0, 1, a.size), a)
        sw_total += np.mean(np.abs(a - b))
    return sw_total / num_projections

def conditional_wasserstein_distance(real_data, gen_data, normalize=False, n_projections=5000, use_ray=False):
    """
    Compute conditional Wasserstein distance (ConditionalWD)
    New parameter:
      use_ray: bool, whether to use Ray for parallelism (default False)
    Returns:
      float: conditional Wasserstein distance
    """
    import numpy as np
    import ot

    # Ensure data is 3D
    real_data = _ensure_3d(real_data)
    gen_data = _ensure_3d(gen_data)

    N, T, D = real_data.shape

    def _compute_one(t):
        real_t = real_data[:, t, :].copy()
        gen_t  = gen_data[:, t, :].copy()
        # Normalize
        if normalize:
            stds = np.std(real_t, axis=0)
            nz   = stds > 0
            real_t[:, nz] /= stds[nz]
            gen_t[:, nz]  /= stds[nz]
            
        # For univariate (D=1), use 1D Wasserstein distance
        if D == 1:
            # Flatten (N,1) to (N,)
            real_flat = real_t.reshape(-1)
            gen_flat = gen_t.reshape(-1)
            return float(ot.wasserstein_1d(real_flat, gen_flat))
        # If exactly 2D, use sliced method for fast approximation
        elif D == 2:
            return sliced_wasserstein_2d(real_t, gen_t,
                                         num_projections=n_projections,
                                         normalize=False,
                                         seed=42)

        # Multivariate EMD (original logic)
        w_r = np.ones(N) / N
        w_g = np.ones(gen_t.shape[0]) / gen_t.shape[0]
        C   = ot.dist(real_t, gen_t, metric='euclidean')
        emd2_val = ot.emd2(w_r, w_g, C)
        return float(np.sqrt(emd2_val))

    # Parallel or serial execution (keep original use_ray branch)
    if use_ray:
        import ray
        ray.init(ignore_reinit_error=True, num_cpus=T)
        real_ref = ray.put(real_data)
        gen_ref  = ray.put(gen_data)

        @ray.remote(num_cpus=1)
        def _ray_task(rd, gd, norm, tt):
            return _compute_one(tt)

        tasks   = [_ray_task.remote(real_ref, gen_ref, normalize, t) for t in range(T)]
        results = ray.get(tasks)
        ray.shutdown()
        return float(sum(results) / T)
    else:
        # Serial execution (can also add tqdm)
        total = sum(_compute_one(t) for t in tqdm(range(T)))
        return float(total / T)

# Frobenius norm difference
def frobenius_corr_diff(real_data, gen_data):
    """Compute Frobenius norm of the difference between correlation matrices"""
    # Process data1
    N1, T, _ = real_data.shape
    X1_flat = real_data[:, :, 0]
    Y1_flat = real_data[:, :, 1]
    flatten_data1 = np.hstack([X1_flat, Y1_flat])
    corr_real = np.corrcoef(flatten_data1, rowvar=False)
    
    # Process data2
    N2, T, _ = gen_data.shape
    X2_flat = gen_data[:, :, 0]
    Y2_flat = gen_data[:, :, 1]
    flatten_data2 = np.hstack([X2_flat, Y2_flat])
    corr_gen = np.corrcoef(flatten_data2, rowvar=False)
    
    diff = corr_real - corr_gen
    return np.linalg.norm(diff, ord='fro')

# Mean squared error
def mse_corr_diff(real_data, gen_data):
    """Compute element-wise mean squared error (MSE) of correlation matrices"""
    # Process data1
    N1, T, _ = real_data.shape
    X1_flat = real_data[:, :, 0]
    Y1_flat = real_data[:, :, 1]
    flatten_data1 = np.hstack([X1_flat, Y1_flat])
    corr_real = np.corrcoef(flatten_data1, rowvar=False)
    
    # Process data2
    N2, T, _ = gen_data.shape
    X2_flat = gen_data[:, :, 0]
    Y2_flat = gen_data[:, :, 1]
    flatten_data2 = np.hstack([X2_flat, Y2_flat])
    corr_gen = np.corrcoef(flatten_data2, rowvar=False)
    
    diff = corr_real - corr_gen
    return np.mean(diff**2)

# Mean absolute error
def mae_corr_diff(real_data, gen_data):
    """Compute element-wise mean absolute error (MAE) of correlation matrices"""
    # Process data1
    N1, T, _ = real_data.shape
    X1_flat = real_data[:, :, 0]
    Y1_flat = real_data[:, :, 1]
    flatten_data1 = np.hstack([X1_flat, Y1_flat])
    corr_real = np.corrcoef(flatten_data1, rowvar=False)
    
    # Process data2
    N2, T, _ = gen_data.shape
    X2_flat = gen_data[:, :, 0]
    Y2_flat = gen_data[:, :, 1]
    flatten_data2 = np.hstack([X2_flat, Y2_flat])
    corr_gen = np.corrcoef(flatten_data2, rowvar=False)
    
    diff = corr_real - corr_gen
    return np.mean(np.abs(diff))

# Correlation coefficient of correlation matrices
def corr_of_corrs(real_data, gen_data):
    """Compute Pearson correlation coefficient between flattened correlation matrices"""
    # Process data1
    N1, T, _ = real_data.shape
    X1_flat = real_data[:, :, 0]
    Y1_flat = real_data[:, :, 1]
    flatten_data1 = np.hstack([X1_flat, Y1_flat])
    corr_real = np.corrcoef(flatten_data1, rowvar=False)
    
    # Process data2
    N2, T, _ = gen_data.shape
    X2_flat = gen_data[:, :, 0]
    Y2_flat = gen_data[:, :, 1]
    flatten_data2 = np.hstack([X2_flat, Y2_flat])
    corr_gen = np.corrcoef(flatten_data2, rowvar=False)
    
    # Flatten to vectors
    vec_real = corr_real.flatten()
    vec_gen = corr_gen.flatten()
    
    # Compute Pearson correlation coefficient
    r = np.corrcoef(vec_real, vec_gen)[0, 1]
    return r

# KL divergence (Kullback-Leibler divergence)
def kl_divergence(real_data, gen_data, bins=50):
    """Compute KL divergence between two distributions"""
    # Flatten data to 1D
    real_x = real_data[:, :, 0].flatten()
    real_y = real_data[:, :, 1].flatten()
    gen_x = gen_data[:, :, 0].flatten()
    gen_y = gen_data[:, :, 1].flatten()
    
    # Compute KL divergence for X dimension
    hist_real_x, bin_edges = np.histogram(real_x, bins=bins, density=True)
    hist_gen_x, _ = np.histogram(gen_x, bins=bin_edges, density=True)
    # Avoid division by zero
    hist_real_x = np.clip(hist_real_x, 1e-10, None)
    hist_gen_x = np.clip(hist_gen_x, 1e-10, None)
    kl_x = entropy(hist_real_x, hist_gen_x)
    
    # Compute KL divergence for Y dimension
    hist_real_y, bin_edges = np.histogram(real_y, bins=bins, density=True)
    hist_gen_y, _ = np.histogram(gen_y, bins=bin_edges, density=True)
    # Avoid division by zero
    hist_real_y = np.clip(hist_real_y, 1e-10, None)
    hist_gen_y = np.clip(hist_gen_y, 1e-10, None)
    kl_y = entropy(hist_real_y, hist_gen_y)
    
    # Return average KL divergence for X and Y dimensions
    return (kl_x + kl_y) / 2

# Wasserstein distance
def wasserstein_distance1(real_data, gen_data):
    """
    Compute 2-Wasserstein distance (Earth Mover's distance) between two datasets
    
    Warning: For large datasets, computation cost is O(N*M), may be slow or memory intensive
    """
    # Convert 3D data to 2D
    # (N, T, 2) -> (N, T*2) flatten X and Y features of each sample into one row
    #  [x1, y1, x2, y2, ..., xT, yT]
    
    # If you want [x1, x2, ..., xT, y1, y2, ..., yT],
    # transpose first: real_data.transpose(0, 2, 1).reshape(real_data.shape[0], -1)
    
    real_flat = real_data.reshape(real_data.shape[0], -1) 
    # What is the logic here, [x1,y1, x2, y2, ... ,x100,y100] or [x1,x2,...,x100,y1,y2,...,y100]?
    #  [x1, y1, x2, y2, ..., xT, yT]
    gen_flat = gen_data.reshape(gen_data.shape[0], -1)
    
    # Uniform weights
    n = real_flat.shape[0]
    m = gen_flat.shape[0]
    w_real = np.ones(n) / n
    w_gen = np.ones(m) / m
    
    # Cost matrix: pairwise Euclidean distances
    cost_matrix = ot.dist(real_flat, gen_flat, metric='euclidean')
    
    # Earth Mover's Distance^2
    emd2_value = ot.emd2(w_real, w_gen, cost_matrix)
    
    # Return actual Wasserstein distance (sqrt of EMD^2)
    return np.sqrt(emd2_value)

# Wasserstein distance
def wasserstein_distance2(real_data, gen_data):
    """
    Compute 2-Wasserstein distance (Earth Mover's distance) between two datasets
    
    Warning: For large datasets, computation cost is O(N*M), may be slow or memory intensive
    """
    # Convert 3D data to 2D
    # (N, T, 2) -> (N, T*2) flatten X and Y features of each sample into one row

    
    # If you want [x1, x2, ..., xT, y1, y2, ..., yT],
    # transpose first: 
    
    real_flat = real_data.transpose(0, 2, 1).reshape(real_data.shape[0], -1)

    gen_flat = gen_data.reshape(gen_data.shape[0], -1)
    
    # Uniform weights
    n = real_flat.shape[0]
    m = gen_flat.shape[0]
    w_real = np.ones(n) / n
    w_gen = np.ones(m) / m
    
    # Cost matrix: pairwise Euclidean distances
    cost_matrix = ot.dist(real_flat, gen_flat, metric='euclidean')
    
    # Earth Mover's Distance^2
    emd2_value = ot.emd2(w_real, w_gen, cost_matrix)
    
    # Return actual Wasserstein distance (sqrt of EMD^2)
    return np.sqrt(emd2_value)

def compare_datasets(real_data, gen_data, model_name="Generated Model"):
    """
    Compare real data and generated data, compute multiple evaluation metrics
    
    Args:
    real_data: real dataset, shape (N, T, 1) 
    gen_data: generated dataset, shape (M, T, 1)
    model_name: model name, used for print output
    """
    print(f"\n====== {model_name} Evaluation Results ======")
    
    # Ensure data is (N,T,1) shape
    real_data = _ensure_3d(real_data)
    gen_data = _ensure_3d(gen_data)

    # [New metric here] Signature distance
    # Default use level 3
    sig_dist_val = path_signature_distance(real_data, gen_data, level=3)
    
    # For univariate data, correlation matrix simplifies to correlation coefficient
    # Convert (N,T,1) to (N,T)
    real_flat = real_data[:, :, 0]
    gen_flat = gen_data[:, :, 0]
    
    # Compute autocorrelation matrix
    real_autocorr = np.corrcoef(real_flat, rowvar=False)
    gen_autocorr = np.corrcoef(gen_flat, rowvar=False)
    
    # Compute and print evaluation metrics
    fnorm_diff = np.linalg.norm(real_autocorr - gen_autocorr, ord='fro')
    mse_val = np.mean((real_autocorr - gen_autocorr) ** 2)
    mae_val = np.mean(np.abs(real_autocorr - gen_autocorr))
    
    # Compute correlation coefficient of autocorrelation matrices
    corr_val = np.corrcoef(real_autocorr.flatten(), gen_autocorr.flatten())[0, 1]
    
    # Univariate Wasserstein distance
    # Flatten data to 1D array
    real_1d = real_flat.flatten()
    gen_1d = gen_flat.flatten()
    
    # Compute 1D Wasserstein distance
    wd_value = ot.wasserstein_1d(real_1d, gen_1d)
    
    # For univariate data, marginal WD is just normal WD
    marginal_wd = wd_value
    
    # Conditional WD is the same for univariate case
    conditional_wd = conditional_wasserstein_distance(real_data, gen_data)
    
    # KL divergence calculation
    bins = 50
    hist_real, bin_edges = np.histogram(real_1d, bins=bins, density=True)
    hist_gen, _ = np.histogram(gen_1d, bins=bin_edges, density=True)
    # Avoid division by zero
    hist_real = np.clip(hist_real, 1e-10, None)
    hist_gen = np.clip(hist_gen, 1e-10, None)
    kl_value = entropy(hist_real, hist_gen)
    
    print(f"Frobenius norm difference: {fnorm_diff:.6f}")
    print(f"Autocorrelation matrix MSE: {mse_val:.6f}")
    print(f"Autocorrelation matrix MAE: {mae_val:.6f}")
    print(f"Autocorrelation matrix flattened Pearson correlation: {corr_val:.6f}")
    print(f"Wasserstein distance: {wd_value:.6f}")
    print(f"Marginal Wasserstein distance (MarginalWD): {marginal_wd:.6f}")
    print(f"Conditional Wasserstein distance (ConditionalWD): {conditional_wd:.6f}")
    print(f"KL divergence: {kl_value:.6f}")
    if sig_dist_val is not None and not np.isnan(sig_dist_val):
        print(f"Signature distance (level 3): {sig_dist_val:.6f}")
    else:
        print(f"Signature distance (level 3): computation failed or not performed")
    
    return {
        'frobenius_norm': fnorm_diff,
        'mse': mse_val,
        'mae': mae_val,
        'correlation': corr_val,
        'wasserstein': wd_value,
        'marginal_wasserstein': marginal_wd,
        'conditional_wasserstein': conditional_wd,
        'kl_divergence': kl_value,
        'signature_distance': sig_dist_val if sig_dist_val is not None else np.nan,
        'wd_value1': wd_value,
        'wd_value2': wd_value  # For univariate, both methods are the same
    }

# =============== Main function ===============
def main():
    """Main function: load data, compute evaluation metrics, and visualize"""
    print("Start loading datasets...")
    real_data, diffwave_data, adaptive_diffwave_data, diffwave_model_name, adaptive_model_name, calibrated_data, ddpm_data, ddpm_model_name = load_datasets()
    
    # Create directory to save charts
    import os
    save_dir = "evaluation_figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created chart save directory: {save_dir}")
    
    # Format model names to readable form (capitalize first letter)
    # Used for display in charts and output
    diffwave_display_name = diffwave_model_name.capitalize()
    adaptive_display_name = ' '.join(word.capitalize() for word in adaptive_model_name.split('_'))
    calibrated_display_name = 'Calibrated AR-GARCH'
    ddpm_display_name = ddpm_model_name.capitalize()
    
    # For filename format (use underscore instead of space)
    diffwave_file_name = diffwave_display_name
    adaptive_file_name = '_'.join(word.capitalize() for word in adaptive_model_name.split('_'))
    calibrated_file_name = 'calibrated_model'
    ddpm_file_name = ddpm_display_name
    
    # Compute unified data ranges
    print("\nComputing unified visualization ranges...")
    
    # Ensure all data is 3D (N,T,1)
    real_data = _ensure_3d(real_data)
    diffwave_data = _ensure_3d(diffwave_data)
    adaptive_diffwave_data = _ensure_3d(adaptive_diffwave_data)
    calibrated_data = _ensure_3d(calibrated_data)
    ddpm_data = _ensure_3d(ddpm_data)
    
    # Compute unified range for autocorrelation matrices
    corr_matrices = [
        np.corrcoef(real_data[:, :, 0], rowvar=False),
        np.corrcoef(diffwave_data[:, :, 0], rowvar=False),
        np.corrcoef(adaptive_diffwave_data[:, :, 0], rowvar=False),
        np.corrcoef(calibrated_data[:, :, 0], rowvar=False),
        np.corrcoef(ddpm_data[:, :, 0], rowvar=False)
    ]
    corr_min = min(np.min(matrix) for matrix in corr_matrices)
    corr_max = max(np.max(matrix) for matrix in corr_matrices)
    print(f"Unified autocorrelation matrix range: [{corr_min:.4f}, {corr_max:.4f}]")
    
    # Compute unified range for autocovariance matrices
    cov_matrices = [
        np.cov(real_data[:, :, 0], rowvar=False),
        np.cov(diffwave_data[:, :, 0], rowvar=False),
        np.cov(adaptive_diffwave_data[:, :, 0], rowvar=False),
        np.cov(calibrated_data[:, :, 0], rowvar=False),
        np.cov(ddpm_data[:, :, 0], rowvar=False)
    ]
    cov_min = min(np.min(matrix) for matrix in cov_matrices)
    cov_max = max(np.max(matrix) for matrix in cov_matrices)
    print(f"Unified autocovariance matrix range: [{cov_min:.4f}, {cov_max:.4f}]")
    
    # Compute unified range for statistical distribution plots
    # Mean
    means_all = [
        np.mean(real_data[:, :, 0], axis=1),
        np.mean(diffwave_data[:, :, 0], axis=1),
        np.mean(adaptive_diffwave_data[:, :, 0], axis=1),
        np.mean(calibrated_data[:, :, 0], axis=1),
        np.mean(ddpm_data[:, :, 0], axis=1)
    ]
    mean_lim = (min(np.min(x) for x in means_all), max(np.max(x) for x in means_all))
    
    # Variance
    vars_all = [
        np.var(real_data[:, :, 0], axis=1),
        np.var(diffwave_data[:, :, 0], axis=1),
        np.var(adaptive_diffwave_data[:, :, 0], axis=1),
        np.var(calibrated_data[:, :, 0], axis=1),
        np.var(ddpm_data[:, :, 0], axis=1)
    ]
    var_lim = (min(np.min(x) for x in vars_all), max(np.max(x) for x in vars_all))
    
    # Skewness
    skew_all = [
        skew(real_data[:, :, 0], axis=1),
        skew(diffwave_data[:, :, 0], axis=1),
        skew(adaptive_diffwave_data[:, :, 0], axis=1),
        skew(calibrated_data[:, :, 0], axis=1),
        skew(ddpm_data[:, :, 0], axis=1)
    ]
    skew_lim = (min(np.min(x) for x in skew_all), max(np.max(x) for x in skew_all))
    
    # Unified range for statistical distribution plots
    stat_xlims = [mean_lim, var_lim, skew_lim]
    print(f"Unified range for statistical distribution plots computed")
    
    # Compute unified range for data summary plots
    all_data = np.concatenate([
        real_data[:, :, 0].flatten(),
        diffwave_data[:, :, 0].flatten(),
        adaptive_diffwave_data[:, :, 0].flatten(),
        calibrated_data[:, :, 0].flatten(),
        ddpm_data[:, :, 0].flatten()
    ])
    
    data_lim = (np.min(all_data), np.max(all_data))
    summary_xlims = [data_lim, data_lim]
    
    # Basic data statistics, using unified range
    data_summary(real_data, "Real Data", save_dir=save_dir, xlims=summary_xlims)
    data_summary(diffwave_data, f"{diffwave_file_name} Data", save_dir=save_dir, xlims=summary_xlims)
    data_summary(adaptive_diffwave_data, f"{adaptive_file_name} Data", save_dir=save_dir, xlims=summary_xlims)
    data_summary(calibrated_data, f"{calibrated_file_name} Data", save_dir=save_dir, xlims=summary_xlims)
    data_summary(ddpm_data, f"{ddpm_file_name} Data", save_dir=save_dir, xlims=summary_xlims)
    
    # Merge all dataset summary plots
    summary_paths = [
        f"{save_dir}/Real_Data_summary.png",
        f"{save_dir}/{diffwave_file_name}_Data_summary.png",
        f"{save_dir}/{adaptive_file_name}_Data_summary.png",
        f"{save_dir}/{calibrated_file_name}_Data_summary.png",
        f"{save_dir}/{ddpm_file_name}_Data_summary.png"
    ]
    merge_images(summary_paths, f"{save_dir}/merged_data_summaries.png")
    print(f"All dataset summary plots merged and saved to: {save_dir}/merged_data_summaries.png")
    
    # Visualize correlation matrices, using unified range
    print("\nPlotting correlation matrices...")
    real_corr_path = f"{save_dir}/real_corr_matrix.png"
    diffwave_corr_path = f"{save_dir}/{diffwave_model_name}_corr_matrix.png"
    adaptive_corr_path = f"{save_dir}/{adaptive_model_name}_corr_matrix.png"
    calibrated_corr_path = f"{save_dir}/calibrated_model_corr_matrix.png"
    ddpm_corr_path = f"{save_dir}/{ddpm_model_name}_corr_matrix.png"
    
    plot_correlation_matrix(real_data, "Real Data Correlation Matrix", real_corr_path, vmin=corr_min, vmax=corr_max)
    plot_correlation_matrix(diffwave_data, f"{diffwave_display_name} Data Correlation Matrix", diffwave_corr_path, vmin=corr_min, vmax=corr_max)
    plot_correlation_matrix(adaptive_diffwave_data, f"{adaptive_display_name} Data Correlation Matrix", adaptive_corr_path, vmin=corr_min, vmax=corr_max)
    plot_correlation_matrix(calibrated_data, f"{calibrated_display_name} Data Correlation Matrix", calibrated_corr_path, vmin=corr_min, vmax=corr_max)
    plot_correlation_matrix(ddpm_data, f"{ddpm_display_name} Data Correlation Matrix", ddpm_corr_path, vmin=corr_min, vmax=corr_max)
    
    # Merge correlation matrix plots
    corr_paths = [real_corr_path, diffwave_corr_path, adaptive_corr_path, calibrated_corr_path, ddpm_corr_path]
    merge_images(corr_paths, f"{save_dir}/merged_correlation_matrices.png")
    
    # Visualize covariance matrices, using unified range
    print("\nPlotting covariance matrices...")
    real_cov_path = f"{save_dir}/real_cov_matrix.png"
    diffwave_cov_path = f"{save_dir}/{diffwave_model_name}_cov_matrix.png"
    adaptive_cov_path = f"{save_dir}/{adaptive_model_name}_cov_matrix.png"
    calibrated_cov_path = f"{save_dir}/calibrated_model_cov_matrix.png"
    ddpm_cov_path = f"{save_dir}/{ddpm_model_name}_cov_matrix.png"
    
    plot_covariance_matrix(real_data, "Real Data Covariance Matrix", real_cov_path, vmin=cov_min, vmax=cov_max)
    plot_covariance_matrix(diffwave_data, f"{diffwave_display_name} Data Covariance Matrix", diffwave_cov_path, vmin=cov_min, vmax=cov_max)
    plot_covariance_matrix(adaptive_diffwave_data, f"{adaptive_display_name} Data Covariance Matrix", adaptive_cov_path, vmin=cov_min, vmax=cov_max)
    plot_covariance_matrix(calibrated_data, f"{calibrated_display_name} Data Covariance Matrix", calibrated_cov_path, vmin=cov_min, vmax=cov_max)
    plot_covariance_matrix(ddpm_data, f"{ddpm_display_name} Data Covariance Matrix", ddpm_cov_path, vmin=cov_min, vmax=cov_max)
    
    # Merge covariance matrix plots
    cov_paths = [real_cov_path, diffwave_cov_path, adaptive_cov_path, calibrated_cov_path, ddpm_cov_path]
    merge_images(cov_paths, f"{save_dir}/merged_covariance_matrices.png")
    
    # Visualize statistical distributions, using unified range
    print("\nPlotting statistical distributions...")
    real_stats_path = f"{save_dir}/real_stats_dist.png"
    diffwave_stats_path = f"{save_dir}/{diffwave_model_name}_stats_dist.png"
    adaptive_stats_path = f"{save_dir}/{adaptive_model_name}_stats_dist.png"
    calibrated_stats_path = f"{save_dir}/calibrated_model_stats_dist.png"
    ddpm_stats_path = f"{save_dir}/{ddpm_model_name}_stats_dist.png"
    
    plot_statistical_distributions(real_data, "Real Data Statistical Distributions", real_stats_path, xlims=stat_xlims)
    plot_statistical_distributions(diffwave_data, f"{diffwave_display_name} Data Statistical Distributions", diffwave_stats_path, xlims=stat_xlims)
    plot_statistical_distributions(adaptive_diffwave_data, f"{adaptive_display_name} Data Statistical Distributions", adaptive_stats_path, xlims=stat_xlims)
    plot_statistical_distributions(calibrated_data, f"{calibrated_display_name} Data Statistical Distributions", calibrated_stats_path, xlims=stat_xlims)
    plot_statistical_distributions(ddpm_data, f"{ddpm_display_name} Data Statistical Distributions", ddpm_stats_path, xlims=stat_xlims)
    
    # Merge statistical distribution plots
    stats_paths = [real_stats_path, diffwave_stats_path, adaptive_stats_path, calibrated_stats_path, ddpm_stats_path]
    merge_images(stats_paths, f"{save_dir}/merged_statistical_distributions.png")
    
    # Compute evaluation metrics
    print("\nComputing evaluation metrics...")
    diffwave_metrics = compare_datasets(real_data, diffwave_data, f"{diffwave_display_name} Model")
    adaptive_metrics = compare_datasets(real_data, adaptive_diffwave_data, f"{adaptive_display_name} Model")
    calibrated_metrics = compare_datasets(real_data, calibrated_data, f"{calibrated_display_name} Model")
    ddpm_metrics = compare_datasets(real_data, ddpm_data, f"{ddpm_display_name} Model")
    
    # Compare performance of models
    print("\n====== Model Performance Comparison ======")
    metrics = ['frobenius_norm', 'mse', 'mae', 'correlation', 'wasserstein', 
              'marginal_wasserstein', 'conditional_wasserstein', 'kl_divergence',
              'signature_distance']
    better_counts = {diffwave_display_name: 0, adaptive_display_name: 0, calibrated_display_name: 0, ddpm_display_name: 0}
    
    # Create performance comparison chart
    fig, axs = plt.subplots(5, 2, figsize=(16, 30))
    bar_width = 0.18

    # Define an offset list for each model
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

    # Helper function to plot a metric for all models
    def plot_metric(ax, metric_name, metrics_dict, display_names, colors):
        index = np.arange(1)
        values = [metrics_dict[name].get(metric_name, np.nan) for name in display_names]
        for i, val in enumerate(values):
            if not np.isnan(val):
                ax.bar(index + offsets[i], val, bar_width, label=display_names[i], color=colors[i], alpha=0.7)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title(f'{metric_name.replace("_", " ").capitalize()} Comparison')
        ax.set_xticks(index)
        ax.set_xticklabels([metric_name.replace("_", " ").capitalize()])
        ax.legend()

    model_display_names = [diffwave_display_name, adaptive_display_name, calibrated_display_name, ddpm_display_name]
    model_metrics = {
        diffwave_display_name: diffwave_metrics,
        adaptive_display_name: adaptive_metrics,
        calibrated_display_name: calibrated_metrics,
        ddpm_display_name: ddpm_metrics
    }
    model_colors = ['blue', 'red', 'green', 'purple']

    # Plotting each metric
    plot_metric(axs[0, 0], 'frobenius_norm', model_metrics, model_display_names, model_colors)
    plot_metric(axs[0, 1], 'mse', model_metrics, model_display_names, model_colors)
    plot_metric(axs[1, 0], 'mae', model_metrics, model_display_names, model_colors)
    plot_metric(axs[1, 1], 'correlation', model_metrics, model_display_names, model_colors)
    plot_metric(axs[2, 0], 'wasserstein', model_metrics, model_display_names, model_colors)
    plot_metric(axs[2, 1], 'kl_divergence', model_metrics, model_display_names, model_colors)
    plot_metric(axs[3, 0], 'marginal_wasserstein', model_metrics, model_display_names, model_colors)
    plot_metric(axs[3, 1], 'conditional_wasserstein', model_metrics, model_display_names, model_colors)
    plot_metric(axs[4, 0], 'signature_distance', model_metrics, model_display_names, model_colors)

    # Can hide the last empty subplot (axs[4,1])
    axs[4, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model performance comparison chart saved to: {save_dir}/model_comparison.png")
    
    # Compare performance for each metric
    for metric in metrics:
        # Skip comparison if all values are NaN
        all_nan = all(np.isnan(model_metrics[name].get(metric, np.nan)) for name in model_display_names)
        if all_nan:
            print(f"{metric}: All model values are NaN, skipping comparison.")
            continue

        values = []
        models_for_metric = []
        for name in model_display_names:
            val = model_metrics[name].get(metric, np.nan)
            if not np.isnan(val):
                values.append(val)
                models_for_metric.append(name)

        if not values: # If all values were NaN after checking
             print(f"{metric}: All model values are NaN or not computed.")
             continue

        # Determine the best model for the metric
        if metric == 'correlation':  # Higher is better for correlation
            best_val_idx = np.argmax(values)
        else:  # Lower is better for other metrics
            best_val_idx = np.argmin(values)

        best_model = models_for_metric[best_val_idx]
        better_counts[best_model] += 1

        print(f"{metric}: " + ", ".join([f"{name}={model_metrics[name].get(metric, np.nan):.6f}" for name in model_display_names]) + f", Best model: {best_model}")
    
    print(f"\nSummary: " + ", ".join([f"{name} number of better metrics: {better_counts[name]}" for name in model_display_names]))
    
    winner = max(better_counts, key=better_counts.get)
    print(f"Overall best performing model: {winner}")
    
    # Save results as text file
    with open(f"{save_dir}/evaluation_results.txt", "w") as f:
        f.write("====== Model Performance Evaluation Results ======\n\n")
        for name in model_display_names:
            f.write(f"{name} Model Evaluation Results:\n")
            for metric, value in model_metrics[name].items():
                f.write(f"{metric}: {value:.6f}\n")
            f.write("\n")

        f.write(f"Summary: " + ", ".join([f"{name} number of better metrics: {better_counts[name]}" for name in model_display_names]) + "\n")
        f.write(f"Overall best performing model: {winner}\n")
    
    print(f"Evaluation results saved to: {save_dir}/evaluation_results.txt")

if __name__ == "__main__":
    main()