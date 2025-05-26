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
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from univariate.diffwave import (
    N, T, DATA_PATH,
)

# Command line argument parsing
parser = argparse.ArgumentParser(description='Adaptive DDPM Regularity Test')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
args = parser.parse_args()

# Set random seed
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("N: ", N )
print("T: ", T)
print("DATA_PATH: ", DATA_PATH)
print(f"SEED: ", SEED)

# ======================
# Regularization Scheme Selection
# ======================
print("\nPlease select regularization scheme:")
print("(A) Off-diagonal Frobenius - Only shrink correlations, no effect on variance")
print("(B) Off-diagonal Frobenius + log-det + τ tr - Enhanced numerical stability") 
print("(C) Exponential distance-weighted L2 - Time series prior (distant correlation decay)")
print("(D) Full Frobenius norm - Complete Frobenius regularization (original method)")
print("\nEnter A, B, C or D: ", end='')

REGULARIZATION_CHOICE = input().upper().strip()
while REGULARIZATION_CHOICE not in ['A', 'B', 'C', 'D']:
    print("Invalid choice, please enter A, B, C or D: ", end='')
    REGULARIZATION_CHOICE = input().upper().strip()

print(f"\nSelected scheme {REGULARIZATION_CHOICE}")

# Set hyperparameters based on choice
if REGULARIZATION_CHOICE == 'A':
    # Off-diagonal Frobenius
    LAMBDA_OFF = 1e-3 / T  # According to suggested 10^-3/d
    LAMBDA_1 = LAMBDA_OFF
    LAMBDA_2 = 0
    print(f"Using parameters: λ_off = {LAMBDA_OFF:.6f}")
elif REGULARIZATION_CHOICE == 'B':
    # Off-diagonal Frobenius + log-det + τ tr
    LAMBDA_OFF = 1e-3 / T
    TAU = 1e-2
    LAMBDA_1 = LAMBDA_OFF
    LAMBDA_2 = TAU
    print(f"Using parameters: λ_off = {LAMBDA_OFF:.6f}, τ = {TAU:.6f}")
elif REGULARIZATION_CHOICE == 'C':
    # Exponential distance-weighted L2
    BETA = 0.2
    LAMBDA_D = 1e-4
    LAMBDA_1 = LAMBDA_D
    LAMBDA_2 = 0
    print(f"Using parameters: β = {BETA:.2f}, λ_d = {LAMBDA_D:.6f}")
elif REGULARIZATION_CHOICE == 'D':
    # Full Frobenius norm
    LAMBDA_1 = 1e-3  # Keep original 1e-3 parameter unchanged
    LAMBDA_2 = 0
    print(f"Using parameters: λ_frobenius = {LAMBDA_1:.6f}")


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Time steps and sample size
# T = 100
# # T = 500
# N = 100 * 1000

# BATCH_SZIE = 64
# # Increase LAMBDA_1 value to improve training process stability
# LAMBDA_1 = 5e-3  # Increase from 1e-3 to 5e-3
# LAMBDA_2 = 0


# LAMBDA_1 = 1
# LAMBDA_1 = 3e-1
# LAMBDA_1 = 1e-1
# LAMBDA_1 = 1e-2
# LAMBDA_1 = 1e-3
# LAMBDA_1 = 5e-3
# LAMBDA_1 = 5e-4
# LAMBDA_2 = 0
# BATCH_SZIE = 128
# BATCH_SZIE = 1024
BATCH_SZIE = 10_000


LR_START = 1e-4
LR_END = 1e-6

NUM_EPOCHS = 1550
# NUM_EPOCHS = 5000
print(f"adaptive_ddpm: NUM_EPOCHS: {NUM_EPOCHS}, LR_START: {LR_START}, LR_END: {LR_END}")


print(f"adaptive_ddpm: LAMBDA_1: {LAMBDA_1}, LAMBDA_2: {LAMBDA_2}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ======================
# Adaptive DDPM Model Definition
# ======================
class AdaptiveDDPM(nn.Module):
    def __init__(self, T):
        super(AdaptiveDDPM, self).__init__()
        self.T = T
        self.embedding_dim = 32

        # Time embedding
        self.time_embed = nn.Embedding(T, self.embedding_dim)

        # Learnable Cholesky factor L (lower triangular matrix)
        self.L_params = nn.Parameter(torch.randn(T, T) * 0.01)
        self.register_buffer('tril_mask', torch.tril(torch.ones(T, T)))

        # Main network structure
        self.model = nn.Sequential(
            nn.Linear(T + self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, T)
        )

    def get_L(self):
        """Generate lower triangular Cholesky factor matrix, ensuring numerical stability"""
        L = self.L_params * self.tril_mask
        diag_indices = torch.arange(self.T)
        
        # Originally used softplus to ensure diagonal elements are positive
        # L[diag_indices, diag_indices] = F.softplus(L[diag_indices, diag_indices])
        
        # Modified: ensure diagonal elements have sufficiently large positive values to avoid numerical instability
        min_diag_value = 0.01  # Set a minimum value
        diag_values = F.softplus(L[diag_indices, diag_indices])
        L[diag_indices, diag_indices] = torch.max(diag_values, torch.ones_like(diag_values) * min_diag_value)
        
        return L

    def get_noise(self, batch_size):
        """Generate correlated noise"""
        xi = torch.randn(batch_size, self.T, device=self.L_params.device)
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
    # Noise scheduling parameters
    betas = torch.linspace(1e-4, 0.02, T, device=device)
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
    # Data preprocessing
    quantile = np.quantile(volumes, 0.99)
    exceeding_mask = volumes > quantile
    rows_to_remove = np.any(exceeding_mask, axis=1)
    valid_rows = ~rows_to_remove
    volumes_filtered = volumes[valid_rows][:int(N*1.2), :T]

    # Data splitting
    S_true = volumes_filtered
    total_samples = len(S_true)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    # Standardization processing
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
    
    # Regularization coefficient
    lambda_1 = LAMBDA_1  # Frobenius regularization
    lambda_2 = LAMBDA_2  # Trace regularization
    
    # TensorBoard recording
    writer = SummaryWriter(f'runs/adaptive_ddpm_{config["time_stamp"]}')
    best_val_loss = float('inf')
    
    # Precompute parameters
    betas = config['betas']
    alphas = config['alphas']
    alpha_bars = config['alpha_bars']

    # Model save path
    model_save_path = f"best_model_adaptive_ddpm_{REGULARIZATION_CHOICE}_{config['time_stamp']}.pth"
    last_model_save_path = f"last_model_adaptive_ddpm_{REGULARIZATION_CHOICE}_{config['time_stamp']}.pth"

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = train_recon = train_reg = 0
        
        for batch in train_loader:
            x0 = batch[0].to(device)
            batch_size = x0.size(0)
            
            # Random time step
            t = torch.randint(0, T, (batch_size,), device=device)
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
            
            # Calculate regularization loss based on selected regularization scheme
            if REGULARIZATION_CHOICE == 'A':
                # Off-diagonal Frobenius norm
                L_off_diag = L - torch.diag(torch.diag(L))
                reg_loss = lambda_1 * torch.sum(L_off_diag**2)
            elif REGULARIZATION_CHOICE == 'B':
                # Off-diagonal Frobenius + log-det + τ tr
                L_off_diag = L - torch.diag(torch.diag(L))
                off_diag_reg = lambda_1 * torch.sum(L_off_diag**2)
                
                # Calculate log-det(Sigma) = 2 * sum(log(diag(L)))
                log_det_sigma = 2 * torch.sum(torch.log(torch.diag(L)))
                
                # Calculate tr(Sigma) = ||L||_F^2
                tr_sigma = torch.sum(L**2)
                
                reg_loss = off_diag_reg - log_det_sigma + lambda_2 * tr_sigma
            elif REGULARIZATION_CHOICE == 'C':
                # Exponential distance-weighted L2
                i_indices = torch.arange(T, device=device).unsqueeze(1)
                j_indices = torch.arange(T, device=device).unsqueeze(0)
                distance_matrix = torch.abs(i_indices - j_indices).float()
                weight_matrix = torch.exp(BETA * distance_matrix)
                reg_loss = lambda_1 * torch.sum(weight_matrix * L**2)
            elif REGULARIZATION_CHOICE == 'D':
                # Full Frobenius norm
                reg_loss = lambda_1 * torch.sum(L**2)
            else:
                # Default case (should not happen)
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
        
        # Validation stage
        model.eval()
        val_loss = val_recon = val_reg = 0
        with torch.no_grad():
            for batch in val_loader:
                x0 = batch[0].to(device)
                batch_size = x0.size(0)
                t = torch.randint(0, T, (batch_size,), device=device)
                alpha_bar_t = alpha_bars[t].view(-1, 1)
                
                epsilon = model.get_noise(batch_size)
                xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
                epsilon_theta = model(xt, t)
                
                r = epsilon_theta - epsilon
                L = model.get_L()
                z = torch.linalg.solve_triangular(L, r.T, upper=False).T
                
                recon_loss = torch.mean(torch.sum(z**2, dim=1))
                
                # Calculate regularization loss for validation set based on selected regularization scheme
                if REGULARIZATION_CHOICE == 'A':
                    L_off_diag = L - torch.diag(torch.diag(L))
                    reg_loss = lambda_1 * torch.sum(L_off_diag**2)
                elif REGULARIZATION_CHOICE == 'B':
                    L_off_diag = L - torch.diag(torch.diag(L))
                    off_diag_reg = lambda_1 * torch.sum(L_off_diag**2)
                    log_det_sigma = 2 * torch.sum(torch.log(torch.diag(L)))
                    tr_sigma = torch.sum(L**2)
                    reg_loss = off_diag_reg - log_det_sigma + lambda_2 * tr_sigma
                elif REGULARIZATION_CHOICE == 'C':
                    i_indices = torch.arange(T, device=device).unsqueeze(1)
                    j_indices = torch.arange(T, device=device).unsqueeze(0)
                    distance_matrix = torch.abs(i_indices - j_indices).float()
                    weight_matrix = torch.exp(BETA * distance_matrix)
                    reg_loss = lambda_1 * torch.sum(weight_matrix * L**2)
                elif REGULARIZATION_CHOICE == 'D':
                    reg_loss = lambda_1 * torch.sum(L**2)
                else:
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
            # Use CPU tensor to save model to avoid potential CUDA memory issues
            model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(model_state_dict, model_save_path)
            print(f"【Save best model】Epoch {epoch+1}, Validation loss: {current_val_loss:.4f}")
        
        # Periodically save the last model to avoid saving best model failure
        if (epoch+1) % 100 == 0 or (epoch+1) == num_epochs:
            model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(model_state_dict, last_model_save_path)
            print(f"【Save latest model】Epoch {epoch+1}")
        
        # Print progress
        if (epoch+1) % 10 == 0:
            # Add extra monitoring information for scheme B
            extra_info = ""
            if REGULARIZATION_CHOICE == 'B':
                with torch.no_grad():
                    Sigma = L @ L.T
                    try:
                        eigenvals = torch.linalg.eigvalsh(Sigma)
                        min_eig = eigenvals.min().item()
                        max_eig = eigenvals.max().item()
                        extra_info = f" | Min eig: {min_eig:.6f}, Max eig: {max_eig:.2f}"
                    except:
                        extra_info = " | Eigenvalue computation failed"
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} "
                  f"(Recon: {train_recon/len(train_loader):.4f}, Reg: {train_reg/len(train_loader):.4f}) | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} "
                  f"(Recon: {val_recon/len(val_loader):.4f}, Reg: {val_reg/len(val_loader):.4f}) | "
                  f"LR: {current_lr:.2e}{extra_info}")

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
        
        for t in tqdm(reversed(range(T)), desc="Sampling"):
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
# Sampling Generation with Low Rank Noise (New function)
# ======================
def inference_with_low_rank_noise(model, N_test, scaler, config, U_k, Lambda_k_sqrt_diag, k, device):
    """Step 5 variant: Sampling with specified low rank noise generator"""
    # Note: This function does not print ">>> Step 5" title because it is called inside timing loop
    model.eval()
    betas = config['betas']
    alphas = config['alphas']
    alpha_bars = config['alpha_bars']
    T_local = config['T'] # Get T from config

    # Precompute low rank generation matrix
    generator_k = U_k @ Lambda_k_sqrt_diag # (T x k)

    with torch.no_grad():
        # Initial noise (here, we still use model.get_noise() or low rank generation, theoretically x_T impact is not big, but for consistency, also use low rank)
        xi_k_init = torch.randn(N_test, k, device=device)
        x_t = (generator_k @ xi_k_init.T).T # (T x k) @ (k x B) -> T x B -> B x T

        # for t in tqdm(reversed(range(T_local)), desc=f"Sampling (k={k})", leave=False): # leave=False to avoid each k progress bar
        for t in reversed(range(T_local)): # Remove tqdm in timing loop to avoid too much output
            t_batch = torch.full((N_test,), t, device=device)
            epsilon_theta = model(x_t, t_batch) # Model prediction part unchanged

            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]

            if t > 0:
                sigma_t = torch.sqrt(betas[t])
                # --- Use low rank noise generator to generate z ---
                xi_k = torch.randn(N_test, k, device=device)
                z = (generator_k @ xi_k.T).T # (T x k) @ (k x B) -> T x B -> B x T
                # --------------------------
            else:
                sigma_t = z = 0

            # Reverse step (unchanged)
            x_t = (x_t - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t)*epsilon_theta) / torch.sqrt(alpha_t)
            x_t += sigma_t * z

        samples = x_t.cpu().numpy()
        # samples = scaler.inverse_transform(x_t.cpu().numpy()) # If need to inverse normalize
    return samples

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M:%S") + f"_{REGULARIZATION_CHOICE}_seed_{SEED}"

# ======================
# Speed Test
# ======================
def measure_noise_generation_speed(model_path, k_values, device, batch_size=128, num_repeats=10_000):
    """Test noise generation speed with different rank k and calculate cumulative explained variance"""
    print(f">>> Step 6: Test noise generation speed (Repeat {num_repeats} times)")

    # Load model and set to evaluation mode
    model = AdaptiveDDPM(T=T).to(device)
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
        eigenvalues = S  # Since Sigma is symmetric, singular values equal eigenvalues
        eigenvectors = U  # Eigenvectors of symmetric matrix equal left singular vectors
        
        print(f"Eigenvalue statistics: min={eigenvalues.min().item():.6e}, max={eigenvalues.max().item():.6e}")
        # Ensure non-negative, SVD already ensured this
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)  # For safety, set minimum threshold
        
        # Calculate total variance (sum of all eigenvalues)
        total_variance = torch.sum(eigenvalues)
        if total_variance <= 0:
            print("Warning: Total sum of eigenvalues non-positive, cannot calculate explained variance ratio.")
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

    # Warmup
    for _ in range(10):
        xi = torch.randn(batch_size, T, device=device)
        _ = xi @ L.T

    # Official timing
    torch.cuda.synchronize() # Ensure CUDA operations completed
    start_event.record()
    for _ in range(num_repeats):
        xi = torch.randn(batch_size, T, device=device)
        _ = xi @ L.T # Use learned L to generate noise
    end_event.record()
    torch.cuda.synchronize() # Ensure timing accurate
    total_time_full = start_event.elapsed_time(end_event) # Milliseconds

    avg_time_full = total_time_full / num_repeats
    # For Full, explained variance is 100%
    results['full'] = (avg_time_full, 1.0)
    # print(f"  k=Full (T={T}): {avg_time_full:.4f} ms/batch") # Commented out

    # --- Test different k values ---
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    cumulative_variance = 0.0 # For accumulating eigenvalues

    for i, k in enumerate(sorted(k_values)): # Best to calculate cumulative value in order of k
        if k > T:
            print(f"  Skipping k={k} as it exceeds dimension T={T}")
            continue
        if k <= 0:
            print(f"  Skipping k={k} as it's non-positive")
            continue

        # Calculate cumulative explained variance
        # Note: Here we assume k_values is increasing, or we calculate sum each time from scratch
        # For more accurate, we directly take sum of first k eigenvalues
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

        # Official timing
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(num_repeats):
            xi_k = torch.randn(batch_size, k, device=device)
            # Low rank noise generation: (N x k) @ (k x B) -> N x B, then transpose to B x N
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
    """Test complete inference time for specified k value"""
    # print(f"   Test k={k_value} inference time...") # Not printing too much info inside

    model.eval() # Ensure model in evaluation mode
    T_local = config['T']

    # --- Based on k_value, select inference function and prepare parameters ---
    if k_value == 'full':
        inference_func = inference # Use original inference function
        params = (model, N_test, scaler, config)
        desc = "Inference Timing (k=Full)"
    elif isinstance(k_value, int) and 0 < k_value <= T_local:
        k = k_value
        U_k = eigenvectors[:, :k]
        Lambda_k_sqrt_diag = torch.diag(sqrt_eigenvalues[:k])
        inference_func = inference_with_low_rank_noise # Use low rank variant
        params = (model, N_test, scaler, config, U_k, Lambda_k_sqrt_diag, k, device)
        desc = f"Inference Timing (k={k})"
    else:
        print(f"   Invalid k value: {k_value}, skip inference time test.")
        return None

    # --- Warmup ---
    # print(f"     Warmup (k={k_value})...")
    for _ in range(num_warmup):
        _ = inference_func(*params)
    # print(f"     Warmup completed (k={k_value}).")

    # --- Official timing ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time_inference = 0.0

    # print(f"     Start timing {num_repeats} inference times (k={k_value})...")
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
# Matrix Analysis Tool
# ======================
def analyze_model_matrices(model, save_path="matrix_debug"):
    """Analyze and save model matrices for debugging eigenvalue decomposition issues"""
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
        
        # Try calculating eigenvalues to check positive definite
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
    print(f"L matrix diagonal: min={L_stats['diag_min']:.6e}, max={L_stats['diag_max']:.6e}")
    print(f"Sigma symmetric: {Sigma_stats['is_symmetric']}")
    print(f"Sigma positive definite: {Sigma_stats['is_positive_definite']}")
    
    if "condition_number" in Sigma_stats:
        print(f"Sigma condition number: {Sigma_stats['condition_number']:.6e}")
    else:
        print(f"Sigma condition number calculation failed: {Sigma_stats.get('condition_number_error', 'Unknown')}")
        
    if "error" not in eigenvals_stats:
        print(f"Eigenvalues: min={eigenvals_stats['min']:.6e}, max={eigenvals_stats['max']:.6e}")
        print(f"Negative eigenvalue count: {eigenvals_stats['negative_count']}")
        print(f"Near zero eigenvalue count: {eigenvals_stats['zero_count']}")
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
    # Initialize configuration
    config = {
        # 'num_epochs': 500,
        # 'num_epochs': 1250,
        'num_epochs': NUM_EPOCHS,
        'T': T,
        'device': device,
        'time_stamp': get_timestamp()
    }
    
    # Record selected regularization scheme to log
    log_file = f"training_log_{config['time_stamp']}.txt"
    with open(log_file, 'w') as f:
        f.write(f"Adaptive DDPM Training Log - {config['time_stamp']}\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Regularization Scheme: {REGULARIZATION_CHOICE}\n")
        if REGULARIZATION_CHOICE == 'A':
            f.write(f"Off-diagonal Frobenius\n")
            f.write(f"λ_off = {LAMBDA_OFF:.6f}\n")
        elif REGULARIZATION_CHOICE == 'B':
            f.write(f"Off-diagonal Frobenius + log-det + τ tr\n")
            f.write(f"λ_off = {LAMBDA_OFF:.6f}, τ = {TAU:.6f}\n")
        elif REGULARIZATION_CHOICE == 'C':
            f.write(f"Exponential distance-weighted L2\n")
            f.write(f"β = {BETA:.2f}, λ_d = {LAMBDA_D:.6f}\n")
        elif REGULARIZATION_CHOICE == 'D':
            f.write(f"Full Frobenius norm\n")
            f.write(f"λ_frobenius = {LAMBDA_1:.6f}\n")
        f.write(f"=" * 50 + "\n\n")
    
    print(f"Training log saved to: {log_file}")
    
    # Step 1: Parameter setup
    config = setup_parameters(config)
    
    # Step 2: Data loading
    data_loader, scaler, test_size = load_data()
    
    # Step 3: Model initialization
    model = AdaptiveDDPM(T=T).to(device)
    
    # Step 4: Train or load model
    model_save_path = f"best_model_adaptive_ddpm_{REGULARIZATION_CHOICE}_{config['time_stamp']}.pth"
    
    last_model_save_path = f"last_model_adaptive_ddpm_{REGULARIZATION_CHOICE}_{config['time_stamp']}.pth"
    model_loaded = False
    
    # First try to analyze model's initial state
    print("\n>>> Analyze initial model state...")
    analyze_model_matrices(model, save_path="matrix_debug_initial")
    
    if not os.path.exists(model_save_path) and not os.path.exists(last_model_save_path):
        print("Model file does not exist, start training...")
        model = train_adaptive_ddpm(model, data_loader, config['num_epochs'], device, config)
        print(f"Training completed, model saved to {model_save_path}")
        # Analyze model immediately after training to ensure matrix information can be obtained
        print("\n>>> Analyze trained model matrix...")
        analyze_model_matrices(model, save_path="matrix_debug_after_training")
        model_loaded = True
    else:
        # Try to load best model or last saved model
        model_path_to_load = model_save_path if os.path.exists(model_save_path) else last_model_save_path
        print(f"Found trained model {model_path_to_load}, will load it.")
        try:
            # Try to load model
            state_dict = torch.load(model_path_to_load, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model {model_path_to_load}")
            model_loaded = True
            
            # Analyze immediately after loading
            print("\n>>> Analyze loaded model matrix...")
            analyze_model_matrices(model, save_path="matrix_debug_after_loading")
        except Exception as e:
            print(f"Loading model {model_path_to_load} failed: {e}")
            traceback.print_exc()  # Print detailed error stack
            
            # Try to check if model file exists and is valid
            print(f"Check model file: {os.path.exists(model_path_to_load)}, file size: {os.path.getsize(model_path_to_load) if os.path.exists(model_path_to_load) else 'N/A'} bytes")
            
            # Try directly load model parameters
            print("Try directly check model parameters...")
            raw_state_dict = torch.load(model_path_to_load, map_location='cpu')
            print(f"Model dictionary type: {type(raw_state_dict)}")
            if isinstance(raw_state_dict, dict):
                print(f"Model dictionary contains keys: {list(raw_state_dict.keys())}")
                for k, v in raw_state_dict.items():
                    print(f"  {k}: shape={v.shape if hasattr(v, 'shape') else 'N/A'}, type={type(v)}")
            
            # Even if loading fails, still try to analyze using original model
            print("\n>>> Model loading failed, use initialized model to continue analysis...")
            analyze_model_matrices(model, save_path="matrix_debug_fallback")
            
            # Re-train a simplified backup model as alternative
            print("\n>>> Re-train simplified backup model...")
            config['num_epochs'] = min(50, config['num_epochs'])  # Reduce training rounds
            model = train_adaptive_ddpm(model, data_loader, config['num_epochs'], device, config)
            print("\n>>> Analyze backup model matrix...")
            analyze_model_matrices(model, save_path="matrix_debug_fallback_trained")
            model_loaded = True

    # --- Precompute eigenvalue decomposition (if model loaded successfully) ---
    eigenvalues, eigenvectors, sqrt_eigenvalues, total_variance = None, None, None, None
    if model_loaded:
        print("Precompute covariance matrix eigenvalue decomposition...")
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
            
            # Directly use SVD decomposition instead of eigenvalue decomposition
            print("Using SVD decomposition to calculate eigenvalues and eigenvectors...")
        
        # SVD decomposition: Sigma = U S V^T, where S is singular values (square root of eigenvalues)
        U, S, Vh = torch.linalg.svd(Sigma)
        
        # S contains singular values, square to get eigenvalues
        eigenvalues = S  # Since Sigma is symmetric, singular values equal eigenvalues
        eigenvectors = U  # Eigenvectors of symmetric matrix equal left singular vectors
        
        # Print eigenvalue information
        print(f"Eigenvalue statistics: min={eigenvalues.min().item():.6e}, max={eigenvalues.max().item():.6e}")
        print(f"First 10 eigenvalues: {eigenvalues[:10].tolist()}")
        
        # Ensure non-negative, use larger minimum threshold
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        # SVD already sorted in descending order, no need to sort
        
        total_variance = torch.sum(eigenvalues)
        if total_variance <= 0: 
            print("Warning: Total sum of eigenvalues non-positive, use default value 1.0.")
            total_variance = torch.tensor(1.0)
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        print("SVD decomposition completed.")

    # Step 6: Test noise generation speed and explained variance
    if model_loaded:
        noise_test_batch_size = 1024 # <--- Modify noise test batch size
        print(f"\n>>> Start testing noise generation speed (Batch size: {noise_test_batch_size})")
        k_to_test_noise = sorted(list(set([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, T])))
        # Pass new batch_size when calling
        speed_variance_results = measure_noise_generation_speed(
            model_save_path,
            k_to_test_noise,
            device,
            batch_size=noise_test_batch_size # <--- Pass modified value
        )
        # ... (Print noise speed table code unchanged) ...
    else:
        print("Cannot perform noise speed test because model loading or eigenvalue decomposition failed.")

    # Step 7: Test complete inference time for different k
    if model_loaded:
        inference_batch_size = 1024 # <--- Modify inference test batch size
        k_to_test_inference = ['full'] + sorted(list(set([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, T])))

        inference_time_results = {}
        print(f"\n>>> Start testing complete inference time for different k values (Batch size: {inference_batch_size})")

        for k_val in tqdm(k_to_test_inference, desc="Overall Inference Timing"):
            avg_time = measure_inference_time_for_k(
                k_val, model,
                inference_batch_size, # <--- Pass modified value
                scaler, config, device,
                eigenvectors, sqrt_eigenvalues,
                num_repeats=5, num_warmup=1 # <--- Note: B increases inference slower, possibly need to reduce repeat times
            )
            if avg_time is not None:
                # Calculate explained variance
                if k_val == 'full' or k_val == T:
                    explained_variance_ratio = 1.0
                elif isinstance(k_val, int):
                    current_k_eigenvalues = eigenvalues[:k_val]
                    cumulative_variance_k = torch.sum(current_k_eigenvalues[current_k_eigenvalues > 0])
                    explained_variance_ratio = (cumulative_variance_k / total_variance).item()
                else:
                    explained_variance_ratio = 0.0 # Should not happen

                inference_time_results[k_val] = (avg_time, explained_variance_ratio)

        # Print inference time table
        print("\nComplete inference time and cumulative explained variance test results:")
        print("------------------------------------------------------------------------------------------")
        print(f"{'k':<5} | {'Avg Inference Time (s/batch)':<28} | {'Explained Variance (%)':<20}")
        print("------------------------------------------------------------------------------------------")

        # Print results, sorted by k ('full' first)
        sorted_keys = ['full'] + sorted([k for k in inference_time_results if k != 'full'])
        for k in sorted_keys:
            if k in inference_time_results:
                speed_ms, variance_ratio = inference_time_results[k]
                speed_s = speed_ms / 1000.0 # Convert to seconds
                k_str = str(k)
                print(f"{k_str:<5} | {speed_s:<28.4f} | {variance_ratio * 100:<20.2f}")
        print("------------------------------------------------------------------------------------------")

    else:
         print("Cannot perform inference time test because model loading or eigenvalue decomposition failed.")

    # Add new analysis function
    if model_loaded:
        analyze_model_matrices(model)

if __name__ == "__main__":
    run_adaptive_ddpm()
    
    

def generate_latex_table():
    """Generate LaTeX table for regularization comparison results"""
    import numpy as np
    
    # Parsed data from regularity_results.txt
    # Scheme A data
    scheme_A = {
        'mae': [0.226816, 0.225759, 0.225675],
        'signature_distance': [0.378185, 0.357755, 0.418258],
        'marginal_wasserstein': [0.360889, 0.355764, 0.368187],
        'correlation': [0.588150, 0.579885, 0.579211],
        'kl_divergence': [0.491885, 0.459155, 0.524759],
        'conditional_wasserstein': [0.359725, 0.354662, 0.367033],
        'frobenius_norm': [31.676679, 31.633829, 31.599519],
        'wasserstein': [0.360889, 0.355764, 0.368187],
        'mse': [0.100341, 0.100070, 0.099853]
    }
    
    # Scheme B data
    scheme_B = {
        'mae': [0.227818, 0.226792, 0.226834],
        'signature_distance': [0.394879, 0.374523, 0.433417],
        'marginal_wasserstein': [0.369615, 0.364272, 0.376653],
        'correlation': [0.576860, 0.569566, 0.567863],
        'kl_divergence': [0.527722, 0.493145, 0.561456],
        'conditional_wasserstein': [0.368401, 0.363115, 0.375450],
        'frobenius_norm': [31.901450, 31.854767, 31.847292],
        'wasserstein': [0.369615, 0.364272, 0.376653],
        'mse': [0.101770, 0.101473, 0.101425]
    }
    
    # Scheme C data
    scheme_C = {
        'mae': [0.226876, 0.223757, 0.224515],
        'signature_distance': [0.364160, 0.332812, 0.394592],
        'marginal_wasserstein': [0.356989, 0.346386, 0.363915],
        'correlation': [0.578599, 0.590835, 0.588820],
        'kl_divergence': [0.469948, 0.416259, 0.504198],
        'conditional_wasserstein': [0.355827, 0.345248, 0.362751],
        'frobenius_norm': [31.770073, 31.392244, 31.470415],
        'wasserstein': [0.356989, 0.346386, 0.363915],
        'mse': [0.100934, 0.098547, 0.099039]
    }
    
    # Scheme D data
    scheme_D = {
        'mae': [0.226708, 0.225657, 0.225560],
        'signature_distance': [0.377861, 0.357549, 0.417577],
        'marginal_wasserstein': [0.360765, 0.355638, 0.368063],
        'correlation': [0.588682, 0.580544, 0.579900],
        'kl_divergence': [0.491323, 0.458652, 0.524135],
        'conditional_wasserstein': [0.359601, 0.354537, 0.366910],
        'frobenius_norm': [31.661752, 31.618482, 31.583174],
        'wasserstein': [0.360765, 0.355638, 0.368063],
        'mse': [0.100247, 0.099973, 0.099750]
    }
    
    schemes = {'A': scheme_A, 'B': scheme_B, 'C': scheme_C, 'D': scheme_D}
    
    # Calculate statistics
    stats = {}
    for scheme_name, scheme_data in schemes.items():
        stats[scheme_name] = {}
        for metric, values in scheme_data.items():
            stats[scheme_name][metric] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1)
            }
    
    # Define metric information
    metrics_info = {
        'frobenius_norm': ('Frobenius Norm Difference', '$\\downarrow$', 'lower_better'),
        'mse': ('Matrix MSE', '$\\downarrow$', 'lower_better'),
        'mae': ('Matrix MAE', '$\\downarrow$', 'lower_better'),
        'correlation': ('Pearson Correlation', '$\\uparrow$', 'higher_better'),
        'kl_divergence': ('KL Divergence', '$\\downarrow$', 'lower_better'),
        'marginal_wasserstein': ('Marginal Wasserstein Distance', '$\\downarrow$', 'lower_better'),
        'conditional_wasserstein': ('Conditional Wasserstein Distance', '$\\downarrow$', 'lower_better'),
        'signature_distance': ('Signature Distance', '$\\downarrow$', 'lower_better'),
    }
    
    # Generate LaTeX table
    latex_table = """

\\begin{table*}[!h]
    \\centering
    
    \\begin{tabular}{lcccccc}
        \\toprule
        \\textbf{Metric} & $\\uparrow\\!/\\!\\downarrow$ &
        \\textbf{Scheme A} & \\textbf{Scheme B} & \\textbf{Scheme C} & \\textbf{Scheme D} \\\\
        \\midrule"""
    
    # Add row for each metric
    for metric, (display_name, arrow, comparison) in metrics_info.items():
        # Find best scheme
        means = [stats[scheme][metric]['mean'] for scheme in ['A', 'B', 'C', 'D']]
        if comparison == 'lower_better':
            best_idx = np.argmin(means)
        else:
            best_idx = np.argmax(means)
        best_scheme = ['A', 'B', 'C', 'D'][best_idx]
        
        # Build table row
        row = f"\n        {display_name:<30} & {arrow:<15} & "
        
        scheme_cells = []
        for scheme in ['A', 'B', 'C', 'D']:
            mean_val = stats[scheme][metric]['mean']
            std_val = stats[scheme][metric]['std']
            
            if metric in ['correlation']:
                # For correlation, keep 4 decimal places
                cell = f"\\meanvar{{{mean_val:.4f}}}{{{std_val:.4f}}}"
            elif metric in ['frobenius_norm']:
                # For frobenius_norm, keep 2 decimal places
                cell = f"\\meanvar{{{mean_val:.2f}}}{{{std_val:.2f}}}"
            else:
                # Other metrics keep 4 decimal places
                cell = f"\\meanvar{{{mean_val:.4f}}}{{{std_val:.4f}}}"
            
            # If it's the best scheme, add red
            if scheme == best_scheme:
                cell = f"\\textcolor{{red}}{{{cell}}}"
            
            scheme_cells.append(cell)
        
        row += " & ".join(scheme_cells) + " \\\\"
        latex_table += row
    
    latex_table += """
        \\bottomrule
    \\end{tabular}
    
    \\caption{Comparison of regularization schemes for Adaptive DDPM across correlation structure preservation metrics. Scheme A: Off-diagonal Frobenius; Scheme B: Off-diagonal Frobenius + log-det + $\\tau$ tr; Scheme C: Exponential distance-weighted L2; Scheme D: Full Frobenius norm. \\textcolor{red}{Red values} indicate the best-performing scheme for each metric. Arrows indicate whether larger ($\\uparrow$) or smaller ($\\downarrow$) values are preferable. Each entry is reported as \\meanvar{mean}{std}, indicating the mean and standard deviation across 3 runs with different random seeds.}
    \\label{tab:regularization_comparison}
\\end{table*}

% Scheme details:
% A: Off-diagonal Frobenius (λ_off = 1e-3/T ≈ 1e-5)
% B: Off-diagonal Frobenius + log-det + τ tr (λ_off = 1e-3/T, τ = 1e-2)  
% C: Exponential distance-weighted L2 (β = 0.2, λ_d = 1e-4)
% D: Full Frobenius norm (λ_frobenius = 1e-3)
"""
    
    # Append to regularity_results.txt file
    with open('univariate/regularity_results.txt', 'a', encoding='utf-8') as f:
        f.write(latex_table)
    
    print("LaTeX table added to univariate/regularity_results.txt")
    return latex_table

# If directly run this script, generate LaTeX table
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--generate-latex":
    generate_latex_table()