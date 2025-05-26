# from matplotlib.dates import num2epoch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import joblib
import traceback

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from univariate.diffwave import (
#     N, T, DATA_PATH,
# )


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set random seed
np.random.seed(42)
torch.manual_seed(42)
# Set random seed for reproducibility
EVALUATION_MODE = True
if EVALUATION_MODE:
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Time steps and sample size
DIFFUSION_T = 100  # Diffusion process time steps
TIME_SERIES_T = 10000  # Time series length
N = 10 * 1000
DATA_PATH = f"ar1_garch1_1_data_{TIME_SERIES_T}.npy"

PRETRAINED_MODEL_PATH = "best_model_adaptive_ddpm_2025-05-23_04-22:29.pth" # 10000


# BATCH_SZIE = 64
# # 增加 LAMBDA_1 的值以提高训练过程的稳定性
# LAMBDA_1 = 5e-3  # 从1e-3增加到5e-3
# LAMBDA_2 = 0


# LAMBDA_1 = 1
LAMBDA_1 = 3e-1
# LAMBDA_1 = 1e-1
# LAMBDA_1 = 1e-2
# LAMBDA_1 = 1e-3
# LAMBDA_1 = 5e-3
# LAMBDA_1 = 5e-4
LAMBDA_2 = 0
# BATCH_SZIE = 128
# BATCH_SZIE = 1024
BATCH_SZIE = 10_000


LR_START = 1e-4
LR_END = 1e-6

NUM_EPOCHS = 1550
# NUM_EPOCHS = 5000
print(f"adaptive_ddpm: NUM_EPOCHS: {NUM_EPOCHS}, LR_START: {LR_START}, LR_END: {LR_END}")


print(f"adaptive_ddpm: LAMBDA_1: {LAMBDA_1}, LAMBDA_2: {LAMBDA_2}")
print(f"adaptive_ddpm: DIFFUSION_T: {DIFFUSION_T}, TIME_SERIES_T: {TIME_SERIES_T}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ======================
# Adaptive DDPM Model Definition
# ======================
class AdaptiveDDPM(nn.Module):
    def __init__(self, diffusion_T, time_series_T):
        super(AdaptiveDDPM, self).__init__()
        self.diffusion_T = diffusion_T
        self.time_series_T = time_series_T
        self.embedding_dim = 32

        # Time embedding - using diffusion time steps
        self.time_embed = nn.Embedding(diffusion_T, self.embedding_dim)

        # Learnable Cholesky factor L (lower triangular matrix) - using time series length
        self.L_params = nn.Parameter(torch.randn(time_series_T, time_series_T) * 0.01)
        self.register_buffer('tril_mask', torch.tril(torch.ones(time_series_T, time_series_T)))

        # Main network structure - input and output dimensions use time series length
        self.model = nn.Sequential(
            nn.Linear(time_series_T + self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, time_series_T)
        )

    def get_L(self):
        """Generate lower triangular Cholesky factor matrix, ensuring numerical stability"""
        L = self.L_params * self.tril_mask
        diag_indices = torch.arange(self.time_series_T)
        
        # Originally used softplus to ensure diagonal elements are positive
        # L[diag_indices, diag_indices] = F.softplus(L[diag_indices, diag_indices])
        
        # Modified: ensure diagonal elements have sufficiently large positive values to avoid numerical instability
        min_diag_value = 0.01  # Set a minimum value
        diag_values = F.softplus(L[diag_indices, diag_indices])
        L[diag_indices, diag_indices] = torch.max(diag_values, torch.ones_like(diag_values) * min_diag_value)
        
        return L

    def get_noise(self, batch_size):
        """Generate correlated noise"""
        xi = torch.randn(batch_size, self.time_series_T, device=self.L_params.device)
        L = self.get_L()
        return xi @ L.T

    def forward(self, x, t):
        """Forward propagation"""
        t_embed = self.time_embed(t)
        x = torch.cat([x, t_embed], dim=1)
        return self.model(x)

# ======================
# Utility Functions and Configuration
# ======================
class AttrDict(dict):
    """Attribute dictionary"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def setup_parameters(config):
    """Step 1: Parameter setup"""
    print(">>> Step 1: Parameter setup")
    # Noise scheduling parameters - using diffusion time steps
    betas = torch.linspace(1e-4, 0.02, DIFFUSION_T, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    config.update({
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars
    })
    return config

# ======================
# Data Loading
# ======================
def load_data():
    """Step 2: Data loading"""
    print(">>> Step 2: Data loading")
    # volumes = np.load("ar1_garch1_1_data_500.npy", allow_pickle=True)
    volumes = np.load(DATA_PATH)
    print("volumes.shape", volumes.shape)
    # # Data preprocessing
    # quantile = np.quantile(volumes, 0.99)
    # exceeding_mask = volumes > quantile
    # rows_to_remove = np.any(exceeding_mask, axis=1)
    # valid_rows = ~rows_to_remove
    # volumes_filtered = volumes[valid_rows][:int(N*1.2), :TIME_SERIES_T]
    # # Data preprocessing
    volumes_filtered = volumes[:N, :TIME_SERIES_T]
    print("volumes_filtered.shape", volumes_filtered.shape)
    

    # Data splitting
    S_true = volumes_filtered
    total_samples = len(S_true)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    # Standardization
    scaler = StandardScaler()
    train_data = scaler.fit_transform(S_true[:train_size])
    val_data = scaler.transform(S_true[train_size:train_size+val_size])
    test_data = scaler.transform(S_true[train_size+val_size:])

    # Create DataLoader
    batch_size = BATCH_SZIE
    train_loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float32)), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_data, dtype=torch.float32)),
                           batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_data, dtype=torch.float32)),
                            batch_size=batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader), scaler, test_size

# ======================
# Model Training
# ======================
def train_adaptive_ddpm(model, dataloader, num_epochs, device, config):
    """Step 4: Train Adaptive DDPM"""
    print(">>> Step 4: Start training")
    train_loader, val_loader, _ = dataloader
    # optimizer = AdamW(model.parameters(), lr=1e-2)
    # optimizer = AdamW(model.parameters(), lr=1e-3)
    optimizer = AdamW(model.parameters(), lr=LR_START)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=LR_END)
    
    # Regularization coefficients
    lambda_1 = LAMBDA_1  # Frobenius regularization
    lambda_2 = LAMBDA_2  # Trace regularization
    
    # TensorBoard logging
    writer = SummaryWriter(f'runs/adaptive_ddpm_{config["time_stamp"]}')
    best_val_loss = float('inf')
    
    # Pre-compute parameters
    betas = config['betas']
    alphas = config['alphas']
    alpha_bars = config['alpha_bars']

    # Model save paths
    model_save_path = f"best_model_adaptive_ddpm_{config['time_stamp']}.pth"
    last_model_save_path = f"last_model_adaptive_ddpm_{config['time_stamp']}.pth"

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = train_recon = train_reg = 0
        
        for batch in train_loader:
            x0 = batch[0].to(device)
            batch_size = x0.size(0)
            
            # Random time steps - using diffusion time steps
            t = torch.randint(0, DIFFUSION_T, (batch_size,), device=device)
            alpha_bar_t = alpha_bars[t].view(-1, 1)
            
            # Generate correlated noise
            epsilon = model.get_noise(batch_size)
            
            # Forward diffusion
            xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
            
            # Predict noise
            epsilon_theta = model(xt, t)
            
            # Calculate residual
            r = epsilon_theta - epsilon
            L = model.get_L()
            
            # Solve triangular system
            z = torch.linalg.solve_triangular(L, r.T, upper=False).T
            
            # Loss calculation
            recon_loss = torch.mean(torch.sum(z**2, dim=1))
            frobenius_reg = torch.sum(L**2)
            trace_reg = torch.sum(torch.diag(L)**2)
            reg_loss = lambda_1 * frobenius_reg + lambda_2 * trace_reg
            loss = recon_loss + reg_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_reg += reg_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = val_recon = val_reg = 0
        with torch.no_grad():
            for batch in val_loader:
                x0 = batch[0].to(device)
                batch_size = x0.size(0)
                t = torch.randint(0, DIFFUSION_T, (batch_size,), device=device)
                alpha_bar_t = alpha_bars[t].view(-1, 1)
                
                epsilon = model.get_noise(batch_size)
                xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
                epsilon_theta = model(xt, t)
                
                r = epsilon_theta - epsilon
                L = model.get_L()
                z = torch.linalg.solve_triangular(L, r.T, upper=False).T
                
                recon_loss = torch.mean(torch.sum(z**2, dim=1))
                reg_loss = lambda_1 * torch.sum(L**2) + lambda_2 * torch.sum(torch.diag(L)**2)
                loss = recon_loss + reg_loss
                
                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_reg += reg_loss.item()
        
        # Learning rate adjustment
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record metrics
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/train_recon', train_recon/len(train_loader), epoch)
        writer.add_scalar('Loss/train_reg', train_reg/len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss/len(val_loader), epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save best model
        current_val_loss = val_loss/len(val_loader)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            # Use CPU tensors to save model, avoiding potential CUDA memory issues
            model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(model_state_dict, model_save_path)
            print(f"[Save Best Model] Epoch {epoch+1}, Validation Loss: {current_val_loss:.4f}")
        
        # Periodically save the latest model to avoid best model save failure
        if (epoch+1) % 100 == 0 or (epoch+1) == num_epochs:
            model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(model_state_dict, last_model_save_path)
            print(f"[Save Latest Model] Epoch {epoch+1}")
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} "
                  f"(Recon: {train_recon/len(train_loader):.4f}, Reg: {train_reg/len(train_loader):.4f}) | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} "
                  f"(Recon: {val_recon/len(val_loader):.4f}, Reg: {val_reg/len(val_loader):.4f}) | "
                  f"LR: {current_lr:.2e}")

    writer.close()
    return model

# ======================
# Sampling Generation
# ======================
def inference(model, N_test, scaler, config):
    """Step 5: Sampling generation"""
    print(">>> Step 5: Sampling generation")
    model.eval()
    betas = config['betas']
    alphas = config['alphas']
    alpha_bars = config['alpha_bars']
    
    with torch.no_grad():
        # Initial noise
        x_t = model.get_noise(N_test).to(device)
        
        for t in tqdm(reversed(range(DIFFUSION_T)), desc="Sampling"):
            t_batch = torch.full((N_test,), t, device=device)
            epsilon_theta = model(x_t, t_batch)
            
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]
            
            if t > 0:
                sigma_t = torch.sqrt(betas[t])
                z = model.get_noise(N_test).to(device)
            else:
                sigma_t = z = 0
            
            # Reverse step
            x_t = (x_t - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t)*epsilon_theta) / torch.sqrt(alpha_t)
            x_t += sigma_t * z
        
        # samples = scaler.inverse_transform(x_t.cpu().numpy())
        samples = x_t.cpu().numpy()
    return samples

# ======================
# Sampling Generation with Low-rank Noise (New Function)
# ======================
def inference_with_low_rank_noise(model, N_test, scaler, config, U_k, Lambda_k_sqrt_diag, k, device):
    """Step 5 variant: Sampling using specified low-rank noise generator"""
    # Note: This function doesn't print ">>> Step 5" title as it's called within timing loops
    model.eval()
    betas = config['betas']
    alphas = config['alphas']
    alpha_bars = config['alpha_bars']
    diffusion_T_local = DIFFUSION_T # Use global DIFFUSION_T

    # Pre-compute low-rank generation matrix
    generator_k = U_k @ Lambda_k_sqrt_diag # (TIME_SERIES_T x k)

    with torch.no_grad():
        # Initial noise (can still use model.get_noise() or low-rank generation, theoretically x_T has little impact, but for consistency, also use low-rank)
        xi_k_init = torch.randn(N_test, k, device=device)
        x_t = (generator_k @ xi_k_init.T).T # (TIME_SERIES_T x k) @ (k x B) -> TIME_SERIES_T x B -> B x TIME_SERIES_T

        # for t in tqdm(reversed(range(diffusion_T_local)), desc=f"Sampling (k={k})", leave=False): # leave=False avoids leaving a progress bar for each k
        for t in reversed(range(diffusion_T_local)): # Remove tqdm in timing loops to avoid excessive output
            t_batch = torch.full((N_test,), t, device=device)
            epsilon_theta = model(x_t, t_batch) # Model prediction part unchanged

            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]

            if t > 0:
                sigma_t = torch.sqrt(betas[t])
                # --- Use low-rank noise generation for z ---
                xi_k = torch.randn(N_test, k, device=device)
                z = (generator_k @ xi_k.T).T # (TIME_SERIES_T x k) @ (k x B) -> TIME_SERIES_T x B -> B x TIME_SERIES_T
                # --------------------------
            else:
                sigma_t = z = 0

            # Reverse step (unchanged)
            x_t = (x_t - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t)*epsilon_theta) / torch.sqrt(alpha_t)
            x_t += sigma_t * z

        samples = x_t.cpu().numpy()
        # samples = scaler.inverse_transform(x_t.cpu().numpy()) # If denormalization is needed
    return samples

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M:%S")

# ======================
# Speed Test
# ======================
def measure_noise_generation_speed(model_path, k_values, device, batch_size=128, num_repeats=10_000):
    """Test noise generation speed for different k values and calculate cumulative explained variance"""
    print(f">>> Step 6: Test noise generation speed (repeat {num_repeats} times)")

    # Load model and set to evaluation mode
    model = AdaptiveDDPM(diffusion_T=DIFFUSION_T, time_series_T=TIME_SERIES_T).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get learned L matrix and Sigma matrix
    with torch.no_grad():
        L = model.get_L().detach()
        Sigma = L @ L.T

    # Perform SVD decomposition on Sigma
    try:
        # Directly use SVD decomposition
        print(f"Sigma statistics: min={Sigma.min().item():.6e}, max={Sigma.max().item():.6e}")
        print(f"Using SVD decomposition to calculate eigenvalues and eigenvectors...")
        
        # SVD decomposition: Sigma = U S V^T
        U, S, Vh = torch.linalg.svd(Sigma)
        
        # Use singular values as eigenvalues
        eigenvalues = S  # Since Sigma is symmetric, singular values are equal to eigenvalues
        eigenvectors = U  # Eigenvectors of symmetric matrix are left singular vectors
        
        print(f"Eigenvalue statistics: min={eigenvalues.min().item():.6e}, max={eigenvalues.max().item():.6e}")
        # Ensure non-negative, SVD has already ensured this
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)  # For safety, set a minimum threshold
        
        # Calculate total variance (sum of all eigenvalues)
        total_variance = torch.sum(eigenvalues)
        if total_variance <= 0:
            print("Warning: Total sum of eigenvalues is non-positive, cannot calculate explained variance ratio.")
            total_variance = torch.tensor(1.0)  # Avoid division by zero
        
        print("SVD decomposition completed successfully.")

    except Exception as e:
        print(f"SVD decomposition failed: {e}")
        return {}  # Cannot perform test

    results = {}

    # --- Test full L matrix ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time_full = 0.0

    print("Warmup...")
    # Warmup
    for _ in tqdm(range(10), desc="Warmup"):
        xi = torch.randn(batch_size, TIME_SERIES_T, device=device)
        _ = xi @ L.T
    print("Warmup done")
    

    # Start timing
    torch.cuda.synchronize() # Ensure CUDA operations are completed
    start_event.record()
    for _ in tqdm(range(num_repeats), desc="Using learned L to generate noise"):
        xi = torch.randn(batch_size, TIME_SERIES_T, device=device)
        _ = xi @ L.T # Use learned L to generate noise
    end_event.record()
    torch.cuda.synchronize() # Ensure timing is accurate
    total_time_full = start_event.elapsed_time(end_event) # Milliseconds

    avg_time_full = total_time_full / num_repeats
    # For Full, explained variance is 100%
    results['full'] = (avg_time_full, 1.0)
    # print(f"  k=Full (T={T}): {avg_time_full:.4f} ms/batch") # Commented out

    # --- Test different k values ---
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    cumulative_variance = 0.0 # For accumulating eigenvalues

    for i, k in enumerate(tqdm(sorted(k_values), desc="Testing k values")): # Best to calculate cumulative values in order of k
        print(f"Testing k={k}")
        if k > TIME_SERIES_T:
            print(f"  Skipping k={k} as it exceeds dimension T={TIME_SERIES_T}")
            continue
        if k <= 0:
            print(f"  Skipping k={k} as it's non-positive")
            continue

        # Calculate cumulative explained variance
        # Note: Here we assume k_values is increasing, or we calculate sum from scratch each time
        # For more accurate results, we directly take sum of first k eigenvalues
        current_k_eigenvalues = eigenvalues[:k]
        cumulative_variance_k = torch.sum(current_k_eigenvalues[current_k_eigenvalues > 0])
        explained_variance_ratio = (cumulative_variance_k / total_variance).item()

        U_k = eigenvectors[:, :k]
        Lambda_k_sqrt_diag = torch.diag(sqrt_eigenvalues[:k])
        generator_k = U_k @ Lambda_k_sqrt_diag

        total_time_k = 0.0

        # Warmup
        for _ in range(10):
            xi_k = torch.randn(batch_size, k, device=device)
            _ = (generator_k @ xi_k.T).T # (N x k) @ (k x B) -> N x B -> B x N

        # Start timing
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(num_repeats):
            xi_k = torch.randn(batch_size, k, device=device)
            # Low-rank noise generation: (N x k) @ (k x B) -> N x B, then transpose to B x N
            _ = (generator_k @ xi_k.T).T
        end_event.record()
        torch.cuda.synchronize()
        total_time_k = start_event.elapsed_time(end_event) # Milliseconds

        avg_time_k = total_time_k / num_repeats
        results[k] = (avg_time_k, explained_variance_ratio)
        # print(f"  k={k}: {avg_time_k:.4f} ms/batch") # Commented out

    return results

# ======================
# Inference Time Test (Modified)
# ======================
# Rename and modify parameters and logic
def measure_inference_time_for_k(k_value, model, N_test, scaler, config, device,
                                 eigenvectors, sqrt_eigenvalues, # Pass in decomposition results
                                 num_repeats=10, num_warmup=2): # Inference slower, reduce repeat times
    """Test full inference time for specified k value"""
    # print(f"   Testing k={k_value} inference time...") # Don't print too much info inside

    model.eval() # Ensure model is in evaluation mode
    T_local = TIME_SERIES_T

    # --- Based on k_value, select inference function and prepare parameters ---
    if k_value == 'full':
        inference_func = inference # Use original inference function
        params = (model, N_test, scaler, config)
        desc = "Inference Timing (k=Full)"
    elif isinstance(k_value, int) and 0 < k_value <= T_local:
        k = k_value
        U_k = eigenvectors[:, :k]
        Lambda_k_sqrt_diag = torch.diag(sqrt_eigenvalues[:k])
        inference_func = inference_with_low_rank_noise # Use low-rank variant
        params = (model, N_test, scaler, config, U_k, Lambda_k_sqrt_diag, k, device)
        desc = f"Inference Timing (k={k})"
    else:
        print(f"   Invalid k value: {k_value}, skipping inference time test.")
        return None

    # --- Warmup ---
    # print(f"     Performing warmup (k={k_value})...")
    for _ in range(num_warmup):
        _ = inference_func(*params)
    # print(f"     Warmup completed (k={k_value}).")

    # --- Start timing ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time_inference = 0.0

    # print(f"     Starting timing {num_repeats} inference times (k={k_value})...")
    torch.cuda.synchronize()
    start_event.record()
    # Use tqdm to display overall progress instead of showing each inference
    for _ in range(num_repeats):
         _ = inference_func(*params)
    end_event.record()
    torch.cuda.synchronize()
    total_time_inference = start_event.elapsed_time(end_event) # Milliseconds

    avg_time_inference = total_time_inference / num_repeats
    # print(f"     Timing completed (k={k_value}). Average: {avg_time_inference:.2f} ms")

    return avg_time_inference

# ======================
# Matrix Analysis Tools
# ======================
def analyze_model_matrices(model, save_path="matrix_debug"):
    """Analyze and save model matrices for debugging feature decomposition issues"""
    import os
    import numpy as np
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        # Get L matrix
        L = model.get_L().detach().cpu().numpy()
        
        # Generate Sigma matrix
        Sigma = L @ L.T
        
        # Save matrices
        np.save(f"{save_path}/L_matrix.npy", L)
        np.save(f"{save_path}/Sigma_matrix.npy", Sigma)
        
        # Analyze L matrix
        L_diag = np.diag(L)
        L_stats = {
            "min": L.min(),
            "max": L.max(),
            "mean": L.mean(),
            "std": L.std(),
            "diag_min": L_diag.min(),
            "diag_max": L_diag.max(),
            "diag_mean": L_diag.mean()
        }
        
        # Analyze Sigma matrix
        Sigma_diag = np.diag(Sigma)
        is_symmetric = np.allclose(Sigma, Sigma.T)
        
        # Try calculating eigenvalues to check positive definiteness
        try:
            eigenvals = np.linalg.eigvals(Sigma)
            is_pd = np.all(eigenvals > -1e-10)  # Allow small numerical error
            eigenvals_stats = {
                "min": eigenvals.min(),
                "max": eigenvals.max(),
                "negative_count": np.sum(eigenvals < 0),
                "zero_count": np.sum(np.abs(eigenvals) < 1e-10)
            }
        except Exception as e:
            eigenvals_stats = {"error": str(e)}
            is_pd = False
        
        Sigma_stats = {
            "min": Sigma.min(),
            "max": Sigma.max(),
            "mean": Sigma.mean(),
            "std": Sigma.std(),
            "diag_min": Sigma_diag.min(),
            "diag_max": Sigma_diag.max(),
            "is_symmetric": is_symmetric,
            "is_positive_definite": is_pd,
        }
        
        try:
            Sigma_stats["condition_number"] = np.linalg.cond(Sigma)
        except Exception as e:
            Sigma_stats["condition_number_error"] = str(e)
        
        # Save statistics
        with open(f"{save_path}/matrix_stats.txt", "w") as f:
            f.write("===== L matrix statistics =====\n")
            for k, v in L_stats.items():
                f.write(f"{k}: {v}\n")
            f.write("\n===== Sigma matrix statistics =====\n")
            for k, v in Sigma_stats.items():
                f.write(f"{k}: {v}\n")
            f.write("\n===== Eigenvalue statistics =====\n")
            if "error" not in eigenvals_stats:
                for k, v in eigenvals_stats.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write(f"error: {eigenvals_stats['error']}\n")
    
    print(f"Matrix analysis saved to {save_path} directory")
    
    # Print key statistics
    print("\nKey matrix statistics:")
    print(f"L matrix diagonal:  min={L_stats['diag_min']:.6e}, max={L_stats['diag_max']:.6e}")
    print(f"Sigma symmetric: {Sigma_stats['is_symmetric']}")
    print(f"Sigma positive definite: {Sigma_stats['is_positive_definite']}")
    
    if "condition_number" in Sigma_stats:
        print(f"Sigma condition number: {Sigma_stats['condition_number']:.6e}")
    else:
        print(f"Sigma condition number calculation failed: {Sigma_stats.get('condition_number_error', 'Unknown')}")
        
    if "error" not in eigenvals_stats:
        print(f"Eigenvalues: min={eigenvals_stats['min']:.6e}, max={eigenvals_stats['max']:.6e}")
        print(f"Negative eigenvalues count: {eigenvals_stats['negative_count']}")
        print(f"Near-zero eigenvalues count: {eigenvals_stats['zero_count']}")
    else:
        print(f"Eigenvalues calculation failed: {eigenvals_stats['error']}")
    
    # Also check L matrix rank
    try:
        L_rank = np.linalg.matrix_rank(L)
        print(f"L matrix rank: {L_rank}/{L.shape[0]}")
    except Exception as e:
        print(f"L matrix rank calculation failed: {e}")
    
    return L_stats, Sigma_stats, eigenvals_stats if "error" not in eigenvals_stats else None

# ======================
# Main Flow
# ======================
def run_adaptive_ddpm():
    # Directly specify pretrained model path
    pretrained_model_path = PRETRAINED_MODEL_PATH
    
    # Initialize configuration
    config = {
        'num_epochs': NUM_EPOCHS,
        'diffusion_T': DIFFUSION_T,
        'time_series_T': TIME_SERIES_T,
        'device': device,
        'time_stamp': get_timestamp()
    }
    
    # Step 1: Parameter setup
    config = setup_parameters(config)
    
    # Step 2: Data loading
    data_loader, scaler, test_size = load_data()
    
    # Step 3: Model initialization
    model = AdaptiveDDPM(diffusion_T=DIFFUSION_T, time_series_T=TIME_SERIES_T).to(device)
    
    # Directly load specified pretrained model
    model_loaded = False
    
    if os.path.exists(pretrained_model_path):
        print(f"Found pretrained model {pretrained_model_path}, loading...")
        try:
            # Load model parameters
            state_dict = torch.load(pretrained_model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded pretrained model {pretrained_model_path}")
            model_loaded = True
            
            # # Load successfully, immediately analyze
            # print("\n>>> Analyzing loaded model matrices...")
            # analyze_model_matrices(model, save_path="matrix_debug_pretrained")
        except Exception as e:
            print(f"Failed to load pretrained model {pretrained_model_path}: {e}")
            traceback.print_exc()
            print(f"Check model file: exists={os.path.exists(pretrained_model_path)}, file size={os.path.getsize(pretrained_model_path) if os.path.exists(pretrained_model_path) else 'N/A'} bytes")
            return
    else:
        print(f"Pretrained model file {pretrained_model_path} does not exist!")
        return

    # --- Pre-compute feature decomposition (if model loaded successfully) ---
    eigenvalues, eigenvectors, sqrt_eigenvalues, total_variance = None, None, None, None
    if model_loaded:
        print("Pre-computing covariance matrix feature decomposition...")
        with torch.no_grad():
            L = model.get_L().detach()
            # Print L matrix information for analysis
            print(f"L matrix statistics: min={L.min().item():.6e}, max={L.max().item():.6e}, mean={L.mean().item():.6e}, std={L.std().item():.6e}")
            print(f"L diagonal values: min={torch.diag(L).min().item():.6e}, max={torch.diag(L).max().item():.6e}")
            
            # Generate covariance matrix
            Sigma = L @ L.T
            
            # Print Sigma matrix information
            diag_Sigma = torch.diag(Sigma)
            print(f"Sigma diagonal statistics: min={diag_Sigma.min().item():.6e}, max={diag_Sigma.max().item():.6e}")
            
            # Directly use SVD decomposition instead of feature decomposition
            print("Using SVD decomposition to calculate eigenvalues and eigenvectors...")
        
        # SVD decomposition: Sigma = U S V^T, where S is singular values (square root of eigenvalues)
        U, S, Vh = torch.linalg.svd(Sigma)
        
        # S contains singular values, squaring to get eigenvalues
        eigenvalues = S  # Since Sigma is symmetric, singular values are equal to eigenvalues
        eigenvectors = U  # Eigenvectors of symmetric matrix are left singular vectors
        
        # Print eigenvalue information
        print(f"Eigenvalue statistics: min={eigenvalues.min().item():.6e}, max={eigenvalues.max().item():.6e}")
        print(f"First 10 eigenvalues: {eigenvalues[:10].tolist()}")
        
        # Ensure non-negative, use larger minimum threshold
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        # SVD is already sorted in descending order, no need to sort
        
        total_variance = torch.sum(eigenvalues)
        if total_variance <= 0: 
            print("Warning: Total sum of eigenvalues is non-positive, using default value 1.0.")
            total_variance = torch.tensor(1.0)
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        print("SVD decomposition completed.")

    # Step 6: Test noise generation speed and explained variance
    if model_loaded:
        noise_test_batch_size = 1024 # <--- Modify noise test batch size
        print(f"\n>>> Starting test noise generation speed (batch size: {noise_test_batch_size})")
        
        # Based on percentage, calculate k value list
        k_percentages = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]  # 1%, 5%, 10%, ..., 99%
        k_values_calculated = [max(1, int(TIME_SERIES_T * p)) for p in k_percentages]
        # Add full dimension
        k_to_test_noise = sorted(list(set(k_values_calculated + [TIME_SERIES_T])))
        
        print(f"TIME_SERIES_T: {TIME_SERIES_T}")
        print(f"Tested k values (based on percentage): {k_to_test_noise}")
        
        # Pass new batch_size when calling
        speed_variance_results = measure_noise_generation_speed(
            pretrained_model_path,
            k_to_test_noise,
            device,
            batch_size=noise_test_batch_size # <--- Pass modified value
        )
        
        # Print noise generation speed table
        print("\nNoise generation speed and cumulative explained variance test results:")
        print("------------------------------------------------------------------------------------------")
        print(f"{'k':<8} | {'k/T (%)':<8} | {'Avg Speed (ms/batch)':<20} | {'Explained Variance (%)':<20}")
        print("------------------------------------------------------------------------------------------")
        
        # Print results, sorted by k
        if 'full' in speed_variance_results:
            speed_ms, variance_ratio = speed_variance_results['full']
            print(f"{'full':<8} | {'100.0':<8} | {speed_ms:<20.4f} | {variance_ratio * 100:<20.2f}")
        
        sorted_k = sorted([k for k in speed_variance_results if k != 'full'])
        for k in sorted_k:
            speed_ms, variance_ratio = speed_variance_results[k]
            k_percentage = (k / TIME_SERIES_T) * 100
            print(f"{k:<8} | {k_percentage:<8.1f} | {speed_ms:<20.4f} | {variance_ratio * 100:<20.2f}")
        print("------------------------------------------------------------------------------------------")
    else:
        print("Cannot perform noise speed test because model loading or decomposition failed.")

    # Step 7: Test full inference time for different k
    if model_loaded:
        inference_batch_size = 1024 # <--- Modify inference test batch size
        
        # Based on percentage, calculate k value list (can use fewer test points to speed up inference test)
        k_percentages_inference = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]  # Fewer test points
        k_values_calculated_inference = [max(1, int(TIME_SERIES_T * p)) for p in k_percentages_inference]
        k_to_test_inference = ['full'] + sorted(list(set(k_values_calculated_inference + [TIME_SERIES_T])))

        inference_time_results = {}
        print(f"\n>>> Starting test full inference time for different k values (batch size: {inference_batch_size})")
        print(f"Tested k values (based on percentage): {[k for k in k_to_test_inference if k != 'full']}")

        for k_val in tqdm(k_to_test_inference, desc="Overall Inference Timing"):
            avg_time = measure_inference_time_for_k(
                k_val, model,
                inference_batch_size, # <--- Pass modified value
                scaler, config, device,
                eigenvectors, sqrt_eigenvalues,
                num_repeats=5, num_warmup=1 # <--- Note: B increases, inference slower, may need to reduce repeat times
            )
            if avg_time is not None:
                # Calculate explained variance
                if k_val == 'full' or k_val == TIME_SERIES_T:
                    explained_variance_ratio = 1.0
                elif isinstance(k_val, int):
                    current_k_eigenvalues = eigenvalues[:k_val]
                    cumulative_variance_k = torch.sum(current_k_eigenvalues[current_k_eigenvalues > 0])
                    explained_variance_ratio = (cumulative_variance_k / total_variance).item()
                else:
                    explained_variance_ratio = 0.0 # Should not happen

                inference_time_results[k_val] = (avg_time, explained_variance_ratio)

        # Print inference time table
        print("\nFull inference time and cumulative explained variance test results:")
        print("------------------------------------------------------------------")
        print(f"{'k':<8} | {'k/T (%)':<8} | {'Avg Inference Time (s/batch)':<28} | {'Explained Variance (%)':<20}")
        print("------------------------------------------------------------------")

        # Print results, sorted by k
        sorted_keys = ['full'] + sorted([k for k in inference_time_results if k != 'full'])
        for k in sorted_keys:
            if k in inference_time_results:
                speed_ms, variance_ratio = inference_time_results[k]
                speed_s = speed_ms / 1000.0 # Convert to seconds
                if k == 'full':
                    k_str = 'full'
                    k_percentage_str = '100.0'
                else:
                    k_str = str(k)
                    k_percentage_str = f"{(k / TIME_SERIES_T) * 100:.1f}"
                print(f"{k_str:<8} | {k_percentage_str:<8} | {speed_s:<28.4f} | {variance_ratio * 100:<20.2f}")
        print("------------------------------------------------------------------")

    else:
         print("Cannot perform inference time test because model loading or decomposition failed.")

    # Add new analysis function
    if model_loaded:
        analyze_model_matrices(model)
    
    print(f"\n>>> Test completed! Used pretrained model: {pretrained_model_path}")

if __name__ == "__main__":
    run_adaptive_ddpm()
    
    