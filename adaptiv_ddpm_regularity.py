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
from diffwave import load_data
# ---------------------------------- #A5
import ot  # For Wasserstein distance (pip install pot)
# ---------------------------------- #A5
import logging

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Time steps and sample size
# ---------------------------------- #A20
T = 100
N = 100 * 1000
# Read hyperparameters from environment variables or use default values
DIFFWAVE_LR = float(os.environ.get("HP_DIFFWAVE_LR", "1e-3"))
DIFFWAVE_LR_ETAMIN = float(os.environ.get("HP_DIFFWAVE_LR_ETAMIN", "1e-5"))
BATCH_SIZE = int(os.environ.get("HP_BATCH_SIZE", "1024"))
# BATCH_SIZE = int(os.environ.get("HP_BATCH_SIZE", "64"))
# DIFFWAVE_LR = 1e-6
# DIFFWAVE_LR = 5e-5
print(f"DIFFWAVE_LR: {DIFFWAVE_LR}")

# BATCH_SIZE = 2048
print(f"BATCH_SIZE: {BATCH_SIZE}")
# ---------------------------------- #A20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ======================
# Adaptive DDPM Model Definition
# ======================
class AdaptiveDDPM(nn.Module):
    # ---------------------------------- #A3
    # Add D=2 for dual-channel, input_dim = T * D = 2T
    # L_params becomes shape (2T, 2T)
    # ---------------------------------- #A3
    def __init__(self, T, D=2):
        super(AdaptiveDDPM, self).__init__()
        self.T = T
        # ---------------------------------- #A3
        self.D = D  # Number of channels
        self.input_dim = T * D
        # ---------------------------------- #A3
        self.embedding_dim = 32
        # TODO This value is also a hyperparameter and can be tuned

        # Time embedding
        self.time_embed = nn.Embedding(T, self.embedding_dim)

        # ---------------------------------- #A3
        # Learnable Cholesky factor L (lower triangular matrix), shape (2T, 2T)
        # ---------------------------------- #A3
        self.L_params = nn.Parameter(torch.randn(self.input_dim, self.input_dim) * 0.01)
        self.register_buffer('tril_mask', torch.tril(torch.ones(self.input_dim, self.input_dim)))

        # Main network structure, input is 2T+embedding_dim, output is 2T
        self.model = nn.Sequential(
            nn.Linear(self.input_dim + self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim)
        )

    def get_L(self):
        """Generate lower triangular Cholesky factor matrix"""
        L = self.L_params * self.tril_mask
        diag_indices = torch.arange(self.input_dim)
        L[diag_indices, diag_indices] = F.softplus(L[diag_indices, diag_indices])
        return L

    # ---------------------------------- #A3
    # Generate correlated noise: shape (batch_size, 2T) => reshape => (batch_size, T, 2)
    # ---------------------------------- #A3
    def get_noise(self, batch_size):
        """Generate correlated noise"""
        xi = torch.randn(batch_size, self.input_dim, device=self.L_params.device)
        L = self.get_L()
        noise_flat = xi @ L.T  # shape (batch_size, 2T)
        noise_2d = noise_flat.view(batch_size, self.D, self.T)  # => (batch_size, 2, T)
        noise_2d = noise_2d.permute(0, 2, 1)  # => (batch_size, T, 2)
        return noise_2d

    # ---------------------------------- #A3
    # Forward step:
    #   1) x: shape (batch_size, T, 2)
    #   2) transpose => (batch_size, 2, T)
    #   3) flatten => (batch_size, 2T)
    #   4) concatenate time embedding => model => (batch_size, 2T)
    #   5) reshape => (batch_size, T, 2)
    # ---------------------------------- #A3
    def forward(self, x, t):
        """Forward propagation"""
        # x shape: (batch_size, T, D=2)
        batch_size = x.shape[0]
        # "Block" flatten: all 'a' values first, then all 'b' values
        # => transpose to (batch_size, D, T) => flatten => (batch_size, D*T)
        x = x.permute(0, 2, 1)                     # => (batch_size, 2, T)
        x_flat = x.reshape(batch_size, -1)         # => (batch_size, 2T)
        
        t_embed = self.time_embed(t)               # (batch_size, embedding_dim)
        x_concat = torch.cat([x_flat, t_embed], dim=1)   # => (batch_size, 2T + embedding_dim)
        out_flat = self.model(x_concat)                  # => (batch_size, 2T)
        
        # Reshape back to (batch_size, 2, T) => transpose => (batch_size, T, 2)
        out_2d = out_flat.view(batch_size, self.D, self.T)
        out_2d = out_2d.permute(0, 2, 1)                # => (batch_size, T, 2)
        return out_2d

# ======================
# Utility Functions and Config
# ======================
class AttrDict(dict):
    """Attribute dictionary"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def setup_parameters(config):
    """Step 1: Parameter setup"""
    print(">>> Step 1: Parameter setup")
    # Noise schedule parameters
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    config.update({
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars
    })
    return config

# ---------------------------------- #A5
def evaluate_metrics(real_data, gen_data):
    """
    Compute the following evaluation metrics:
    1. Frobenius norm difference (correlation matrix difference)
    2. MSE (mean squared error of correlation matrix elements)
    3. MAE (mean absolute error of correlation matrix elements)
    4. correlation (Pearson correlation coefficient of flattened correlation matrices)
    5. wasserstein (Wasserstein distance)

    real_data, gen_data: numpy arrays of shape (N, T, 2)
    Returns: (fro_diff, mse_val, mae_val, corr_val, wd_val)
    """
    def corrcoef_2d(data):
        # data shape: (N, T, 2)
        # concatenate => (N, 2T)
        X_flat = data[:, :, 0]
        Y_flat = data[:, :, 1]
        flatten_data = np.hstack([X_flat, Y_flat])
        return np.corrcoef(flatten_data, rowvar=False)

    corr_real = corrcoef_2d(real_data)
    corr_gen = corrcoef_2d(gen_data)

    diff = corr_real - corr_gen
    
    # 1. Frobenius norm difference
    fro_diff = np.linalg.norm(diff, ord='fro')
    # 2. MSE
    mse_val = np.mean(diff**2)
    # 3. MAE
    mae_val = np.mean(np.abs(diff))
    # 4. Correlation coefficient between flattened correlation matrices
    vec_real = corr_real.flatten()
    vec_gen = corr_gen.flatten()
    corr_val = np.corrcoef(vec_real, vec_gen)[0, 1]

    # 5. Wasserstein distance (2-Wasserstein)
    # First reshape (N, T, 2) => (N, T*2), then compute cost matrix of Euclidean distances
    real_flat = real_data.reshape(real_data.shape[0], -1)
    gen_flat = gen_data.reshape(gen_data.shape[0], -1)
    n = real_flat.shape[0]
    m = gen_flat.shape[0]
    w_real = np.ones(n) / n
    w_gen = np.ones(m) / m
    cost_matrix = ot.dist(real_flat, gen_flat, metric='euclidean')
    emd2_value = ot.emd2(w_real, w_gen, cost_matrix)
    wd_val = np.sqrt(emd2_value)

    return fro_diff, mse_val, mae_val, corr_val, wd_val
# ---------------------------------- #A5

# ======================
# Model Training
# ======================
def train_adaptive_ddpm(model, dataloader, num_epochs, device, config, logger):
    """Step 4: Train Adaptive DDPM"""
    print(">>> Step 4: Start training")
    train_loader, val_loader, _ = dataloader
    optimizer = AdamW(model.parameters(), lr=DIFFWAVE_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=DIFFWAVE_LR_ETAMIN)

    # Regularization coefficients, read from environment variables or use default values
    lambda_1 = float(os.environ.get("HP_LAMBDA_1", "1e-3"))  # λ_off (off-diagonal Frobenius)
    lambda_2 = float(os.environ.get("HP_LAMBDA_2", "0"))  # λ_diag (diagonal L2, default 0)
    tau = float(os.environ.get("HP_TAU", "1e-2")) # τ (log-det barrier)
    print(f"Lambda 1 (Off-diag Frobenius λ_off): {lambda_1}, Lambda 2 (Diagonal L2 λ_diag): {lambda_2}, Tau (Barrier τ): {tau}")
    
    
    logger.info(f"HP_LAMBDA_1 (Off-diag Frobenius λ_off): {lambda_1}")
    logger.info(f"HP_LAMBDA_2 (Diagonal L2 λ_diag): {lambda_2}")
    logger.info(f"HP_TAU (Barrier τ): {tau}")
    logger.info(f"HP_DIFFWAVE_LR_ETAMIN: {DIFFWAVE_LR_ETAMIN}")

    # Early stopping parameters, read from environment variables or use default values
    patience = int(os.environ.get("HP_PATIENCE", "100"))
    min_delta = float(os.environ.get("HP_MIN_DELTA", "1e-5"))
    print(f"Early Stopping Patience: {patience}, Min Delta: {min_delta}")

    # TensorBoard logging
    writer = SummaryWriter(f'runs/adaptive_ddpm_{config["run_id"]}') # use run_id
    best_val_loss = float('inf')

    # Early stopping related variables
    epochs_no_improve = 0
    early_stop = False

    # Precompute parameters
    betas = config['betas']
    alphas = config['alphas']
    alpha_bars = config['alpha_bars']

    for epoch in tqdm(range(num_epochs)):
        if early_stop:
            print("Early stopping triggered.")
            break

        model.train()
        train_loss = train_recon = train_reg = 0

        for batch in train_loader:
            # ---------------------------------- #A3
            # x0 shape: (batch_size, T, 2)
            # Use shape (batch_size, T, 2) for forward diffusion
            # ---------------------------------- #A3
            x0 = batch[0].to(device)
            batch_size_ = x0.size(0)

            # Random time steps
            t = torch.randint(0, T, (batch_size_,), device=device)
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1)  # shape (batch_size_,1,1)

            # Generate correlated noise
            epsilon = model.get_noise(batch_size_)      # shape (batch_size_, T, 2)

            # Forward diffusion
            xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon

            # Predict noise
            epsilon_theta = model(xt, t)                # shape (batch_size_, T, 2)

            # Compute residual
            r = epsilon_theta - epsilon  # (batch_size_, T, 2)

            # Flatten r => (batch_size_, 2T)
            r_flat = r.permute(0, 2, 1)  # => (batch_size_, 2, T)
            r_flat = r_flat.reshape(batch_size_, -1)  # => (batch_size_, 2T)

            L = model.get_L()

            # Solve triangular system => shape => (batch_size_, 2T)
            z = torch.linalg.solve_triangular(L, r_flat.T, upper=False).T

            # Loss calculation
            recon_loss = torch.mean(torch.sum(z**2, dim=1))

            # --- New regularization calculation ---
            # 1. Off-diagonal Frobenius norm squared of L
            diag_mask = torch.eye(L.size(0), device=device).bool()
            # Ensure mask is on the same device as L
            if diag_mask.device != L.device:
                diag_mask = diag_mask.to(L.device)

            off_diag_L_sq = L[~diag_mask]**2
            off_diag_frobenius_reg = torch.sum(off_diag_L_sq)

            # 2. log-det and trace barrier
            # Avoid log(0) or negative: F.softplus ensures positive diagonal elements for L
            # The diagonal of L is guaranteed > 0 by F.softplus, so log is safe
            log_diag_L = torch.log(torch.diag(L))
            barrier_term = -2 * torch.sum(log_diag_L) + tau * torch.sum(L**2) # torch.sum(L**2) is ||L||_F^2 = tr(L@L.T)

            # 3. Optional diagonal L2 (controlled by lambda_2)
            diagonal_l2_reg = lambda_2 * torch.sum(torch.diag(L)**2)

            # Total regularization loss: λ_off * off-diagonal Frobenius + Barrier + λ_diag * diagonal L2
            reg_loss = lambda_1 * off_diag_frobenius_reg + barrier_term + diagonal_l2_reg
            # --- End of new regularization calculation ---

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
                batch_size_ = x0.size(0)
                t = torch.randint(0, T, (batch_size_,), device=device)
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1)

                epsilon = model.get_noise(batch_size_)
                xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
                epsilon_theta = model(xt, t)

                r = epsilon_theta - epsilon
                r_flat = r.permute(0, 2, 1)  # => (batch_size_, 2, T)
                r_flat = r_flat.reshape(batch_size_, -1)  # => (batch_size_, 2T)

                L = model.get_L()
                z = torch.linalg.solve_triangular(L, r_flat.T, upper=False).T

                recon_loss = torch.mean(torch.sum(z**2, dim=1))

                # --- New regularization calculation (Validation) ---
                diag_mask = torch.eye(L.size(0), device=device).bool()
                 # Ensure mask is on the same device as L
                if diag_mask.device != L.device:
                    diag_mask = diag_mask.to(L.device)
                off_diag_L_sq = L[~diag_mask]**2
                off_diag_frobenius_reg = torch.sum(off_diag_L_sq)

                log_diag_L = torch.log(torch.diag(L))
                barrier_term = -2 * torch.sum(log_diag_L) + tau * torch.sum(L**2)

                diagonal_l2_reg = lambda_2 * torch.sum(torch.diag(L)**2)

                reg_loss = lambda_1 * off_diag_frobenius_reg + barrier_term + diagonal_l2_reg
                # --- End of new regularization calculation (Validation) ---

                loss = recon_loss + reg_loss

                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_reg += reg_loss.item()

        avg_val_loss = val_loss/len(val_loader)

        # Learning rate adjustment
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/train_recon', train_recon/len(train_loader), epoch)
        writer.add_scalar('Loss/train_reg', train_reg/len(train_loader), epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), f"best_model_adaptive_ddpm_{config['run_id']}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True

        # ---------------------------------- #A5
        # Compute four metrics (using val data and model-generated data of the same size)
        # Collect full val_data
        val_data_list = []
        for batch in val_loader:
            val_data_list.append(batch[0].cpu().numpy())  # (batch_size, T, 2)
        val_data = np.concatenate(val_data_list, axis=0)  # (N_val, T, 2)

        # Use the same number of samples as val_data for inference
        gen_samples = None
        with torch.no_grad():
            # Initial noise (N_val, T, 2)
            x_t = model.get_noise(val_data.shape[0]).to(device)
            for t_ in reversed(range(T)):
                t_batch = torch.full((val_data.shape[0],), t_, device=device)
                epsilon_theta = model(x_t, t_batch)  # (N_val, T, 2)
                alpha_t = alphas[t_]
                alpha_bar_t = alpha_bars[t_]
                if t_ > 0:
                    sigma_t = torch.sqrt(betas[t_])
                    z = model.get_noise(val_data.shape[0]).to(device)
                else:
                    sigma_t = 0
                    z = 0
                x_t = (x_t - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t)*epsilon_theta) / torch.sqrt(alpha_t)
                x_t += sigma_t * z
            gen_samples = x_t.cpu().numpy()  # (N_val, T, 2)

        # Compute evaluation metrics
        # Note: evaluate_metrics uses numpy.corrcoef, which may have numerical issues
        # And this evaluation is performed every epoch, which may be slow
        try:
             fro_diff, mse_val, mae_val, corr_val, wd_val = evaluate_metrics(val_data, gen_samples)
             metrics_str = f" | FN: {fro_diff:.4f}, MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, Corr: {corr_val:.4f}, WD: {wd_val:.4f}"
             writer.add_scalar('Metrics/Frobenius_Diff', fro_diff, epoch)
             writer.add_scalar('Metrics/MSE', mse_val, epoch)
             writer.add_scalar('Metrics/MAE', mae_val, epoch)
             writer.add_scalar('Metrics/Correlation', corr_val, epoch)
             writer.add_scalar('Metrics/Wasserstein_Dist', wd_val, epoch)

        except Exception as e:
             metrics_str = f" | Metric Calculation Error: {e}"
             # Optional: log error to log file
             logging.error(f"Error calculating metrics at epoch {epoch+1}: {e}")
             # If error occurs, do not update tensorboard metrics

        # ---------------------------------- #A5

        print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss/len(train_loader):.4f} "
                f"(Recon: {train_recon/len(train_loader):.4f}, Reg: {train_reg/len(train_loader):.4f}) | "
                f"Val Loss: {avg_val_loss:.4f} "
                f"(Recon: {val_recon/len(val_loader):.4f}, Reg: {val_reg/len(val_loader):.4f}) | "
                f"LR: {current_lr:.2e}"
                f"{metrics_str}") # add metrics_str

        if early_stop:
            break

    writer.close()
    return model

# ======================
# Sampling Generation
# ======================
def inference(model, N_test, scalers, config):
    """Step 5: Sampling generation"""
    print(">>> Step 5: Sampling generation")
    model.eval()
    betas = config['betas']
    alphas = config['alphas']
    alpha_bars = config['alpha_bars']
    
    with torch.no_grad():
        # Initial noise, shape (N_test, T, 2)
        x_t = model.get_noise(N_test).to(device)
        
        for t_ in tqdm(reversed(range(T)), desc="Sampling"):
            t_batch = torch.full((N_test,), t_, device=device)
            epsilon_theta = model(x_t, t_batch)  # (N_test, T, 2)
            
            alpha_t = alphas[t_]
            alpha_bar_t = alpha_bars[t_]
            
            if t_ > 0:
                sigma_t = torch.sqrt(betas[t_])
                z = model.get_noise(N_test).to(device)
            else:
                sigma_t = 0
                z = 0
            
            # Reverse step
            x_t = (x_t - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t)*epsilon_theta) / torch.sqrt(alpha_t)
            x_t += sigma_t * z
        
        samples = x_t.cpu().numpy()

    # Optional per-channel inverse transform:
    # channel_x = samples[:, :, 0].reshape(-1, 1)
    # channel_y = samples[:, :, 1].reshape(-1, 1)
    # x_inv = scalers['x'].inverse_transform(channel_x).reshape(N_test, T)
    # y_inv = scalers['y'].inverse_transform(channel_y).reshape(N_test, T)
    # samples_inversed = np.stack([x_inv, y_inv], axis=-1)
    # return samples_inversed

    return samples


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S") # Use more precise timestamp


# ======================
# Main Process
# ======================
def run_adaptive_ddpm():
    # Initialize config
    run_id = os.environ.get("RUN_ID", get_timestamp()) # Get run_id from environment variable or generate one
    num_epochs = int(os.environ.get("HP_NUM_EPOCHS", "1550")) # Read epochs from environment variable
    # Read early stopping parameters from environment variable or use default values
    patience = int(os.environ.get("HP_PATIENCE", "100"))
    min_delta = float(os.environ.get("HP_MIN_DELTA", "1e-5"))

    # Set up logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"adaptive_ddpm_{run_id}.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler() # Also output to console
                        ])
    logger = logging.getLogger(__name__) # Get logger instance

    config = {
        'num_epochs': num_epochs,
        'T': T,
        'device': device,
        'time_stamp': get_timestamp(), # Still keep a timestamp for internal file names
        'run_id': run_id, # Add run_id to config
        'patience': patience, # Add early stopping parameter to config
        'min_delta': min_delta # Add early stopping parameter to config
    }

    logger.info(f"Starting run with ID: {run_id}")
    logger.info(f"HP_DIFFWAVE_LR: {DIFFWAVE_LR}")
    logger.info(f"HP_BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"Using device: {device}")
    logger.info(f"HP_NUM_EPOCHS: {num_epochs}")
    # logger.info(f"HP_LAMBDA_1: {os.environ.get('HP_LAMBDA_1', '1e-2')}")
    # logger.info(f"HP_LAMBDA_2: {os.environ.get('HP_LAMBDA_2', '1e-2')}")
    logger.info(f"HP_PATIENCE: {patience}")
    logger.info(f"HP_MIN_DELTA: {min_delta}")


    # Step 1: Parameter setup
    logger.info(">>> Step 1: Parameter setup")
    config = setup_parameters(config)

    # Step 2: Data loading (now 2D)
    logger.info(">>> Step 2: Data loading")
    data_loader, scalers, test_size = load_data()

    # Step 3: Initialize model with D=2
    logger.info(">>> Step 3: Initialize model")
    model = AdaptiveDDPM(T=T, D=2).to(device)

    # Step 4: Train model
    logger.info(">>> Step 4: Start training")
    model = train_adaptive_ddpm(model, data_loader, config['num_epochs'], device, config, logger)

    # Step 5: Generate samples
    # After early stopping, we should load the best model weights for inference
    best_model_path = f"best_model_adaptive_ddpm_{config['run_id']}.pth"
    best_model_path = f"best_model_adaptive_ddpm_regularity_{config['run_id']}.pth"
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
         logger.warning(f"Warning: Best model not found at {best_model_path}. Using the model from the last epoch.")

    samples = inference(model, test_size, scalers, config)

    # Save results
    save_dir = f"results_ddpm_{config['run_id']}" # Use run_id as main output directory
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/samples.npy", samples)
    joblib.dump(scalers, f"{save_dir}/scalers.joblib")
    logger.info(f"Results saved to {save_dir}")
    # Print the path of generated samples for evaluation scripts
    logger.info(f"Generated samples path: {os.path.abspath(f'{save_dir}/samples.npy')}")

if __name__ == "__main__":
    run_adaptive_ddpm()