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
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import joblib
    
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from univariate.diffwave import (
    N, T, DATA_PATH,

)

# os.environ["CUDA_VISIBLE_DEVICES"] = "4" # Set GPU (if needed)

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# # Number of time steps and sample size (can be adjusted as needed)
# T = 100
# N = 100 * 1000



LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 500 # 可以根据需要调整训练轮数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ======================
# Standard DDPM Model Definition
# ======================
class DDPM(nn.Module):
    def __init__(self, T_steps):
        super(DDPM, self).__init__()
        self.T_steps = T_steps
        self.embedding_dim = 32

        # 时间嵌入
        self.time_embed = nn.Embedding(T_steps, self.embedding_dim)

        # 主网络结构 (与 adaptive_ddpm.py 类似，但移除了 L 相关部分)
        self.model = nn.Sequential(
            nn.Linear(T_steps + self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, T_steps)
        )

    def forward(self, x, t):
        """前向传播"""
        t_embed = self.time_embed(t)
        x = torch.cat([x, t_embed], dim=1)
        return self.model(x)

# ======================
# Utility Functions and Configuration
# ======================
class AttrDict(dict):
    """Attribute Dictionary"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def setup_parameters(config):
    """Step 1: Parameter Setup"""
    print(">>> Step 1: Parameter Setup")
    T_local = config['T'] # 从 config 获取 T
    # 噪声调度参数
    betas = torch.linspace(1e-4, 0.02, T_local, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    config.update({
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars
    })
    return config

# ======================
# Data Loading (reuse adaptive_ddpm.py logic)
# ======================
def load_data(config):
    """Step 2: Data Loading"""
    print(">>> Step 2: Data Loading")
    T_local = config['T']
    # Make sure this data file exists or replace with your data file path
    # data_path = "ar1_garch1_1_data_500.npy"
    data_path = DATA_PATH
    if not os.path.exists(data_path):
         print(f"Warning: Data file {data_path} not found.")
         print("Generating random data as substitute...")
         volumes = np.random.randn(int(N * 1.2), T_local) # Generate data with similar shape
    else:
         volumes = np.load(data_path, allow_pickle=True)

    print("Original data shape:", volumes.shape)

    # Data preprocessing (take N*1.2 samples, T_local time steps)
    # Ensure data is 2D (num_samples, num_features) where num_features is T_local
    if volumes.ndim == 1:
        volumes = volumes.reshape(-1, 1) # If 1D, adjust if needed
    
    # If data time steps don't match T_local, truncate or pad (simple truncation here)
    if volumes.shape[1] > T_local:
        volumes_filtered = volumes[:int(N*1.2), :T_local]
    elif volumes.shape[1] < T_local:
        print(f"Warning: Data time steps {volumes.shape[1]} less than T_local {T_local}. May need padding or T_local adjustment.")
        # Simply use data's own time steps and update T_local to avoid errors
        # Better approach would be to ensure data meets requirements or pad
        T_local = volumes.shape[1]
        config['T'] = T_local # Update global config T value
        print(f"T updated to {T_local}")
        volumes_filtered = volumes[:int(N*1.2), :]
    else:
        volumes_filtered = volumes[:int(N*1.2), :T_local]
        
    print("Processed data shape:", volumes_filtered.shape)

    # Data splitting
    S_true = volumes_filtered
    total_samples = len(S_true)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    # Standardization processing (directly process 1D data)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(S_true[:train_size])
    val_data = scaler.transform(S_true[train_size:train_size+val_size])
    test_data = scaler.transform(S_true[train_size+val_size:])

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float32)),
                             batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_data, dtype=torch.float32)),
                           batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_data, dtype=torch.float32)),
                            batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader, test_loader), scaler, test_size, config

# ======================
# Model Setup
# ======================
def setup_model(config):
    """Step 3: Set Model"""
    print(">>> Step 3: Set Model")
    model = DDPM(T_steps=config['T']).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    # Use MSELoss (L2 loss) as standard DDPM loss function
    loss_fn = nn.MSELoss()
    print(f"Loss Function: {loss_fn}")
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-7) # Cosine annealing learning rate
    return model, optimizer, scheduler, loss_fn

# ======================
# Model Training
# ======================
def train_ddpm(model, dataloader, optimizer, scheduler, loss_fn, config):
    """Step 4: Train Standard DDPM"""
    print(">>> Step 4: Start Training")
    train_loader, val_loader, _ = dataloader
    T_local = config['T'] # Get T from config
    alpha_bars = config['alpha_bars'] # Get precomputed alpha_bars
    num_epochs = config['num_epochs']

    # TensorBoard recording
    writer = SummaryWriter(f'runs/ddpm_{config["time_stamp"]}')
    best_val_loss = float('inf')

    for epoch in tqdm(range(num_epochs)):
        # --- Training phase ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            x0 = batch[0].to(device) # (batch_size, T_local)
            batch_size = x0.size(0)

            # Random time step
            t = torch.randint(0, T_local, (batch_size,), device=device).long()
            alpha_bar_t = alpha_bars[t].view(-1, 1) # (batch_size, 1)

            # Generate standard Gaussian noise
            epsilon = torch.randn_like(x0) # (batch_size, T_local)

            # Forward diffusion
            xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * epsilon

            # Predict noise
            epsilon_theta = model(xt, t)

            # Calculate standard MSE loss
            loss = loss_fn(epsilon_theta, epsilon)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --- Validation phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x0 = batch[0].to(device)
                batch_size = x0.size(0)
                t = torch.randint(0, T_local, (batch_size,), device=device).long()
                alpha_bar_t = alpha_bars[t].view(-1, 1)

                epsilon = torch.randn_like(x0)
                xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * epsilon
                epsilon_theta = model(xt, t)
                loss = loss_fn(epsilon_theta, epsilon)
                val_loss += loss.item()

        # Learning rate adjustment
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Calculate average loss
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Record metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model_ddpm_{config['time_stamp']}.pth")

        print(f"[DDPM] Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                f"LR: {current_lr:.7f}")

    writer.close()
    return model

# ======================
# Sampling Generation
# ======================
def inference(model, N_test, config):
    """Step 5: Sampling Generation"""
    print(">>> Step 5: Sampling Generation")
    model.eval()
    T_local = config['T']
    alphas = config['alphas']
    alpha_bars = config['alpha_bars']
    betas = config['betas']
    
    with torch.no_grad():
        # Initial noise (standard Gaussian distribution)
        x_t = torch.randn(N_test, T_local, device=device)

        for t_idx in tqdm(reversed(range(T_local)), desc="Sampling"):
            t_batch = torch.full((N_test,), t_idx, device=device).long()
            epsilon_theta = model(x_t, t_batch) # Predict noise

            alpha_t = alphas[t_idx]
            alpha_bar_t = alpha_bars[t_idx]

            if t_idx > 0:
                sigma_t = torch.sqrt(betas[t_idx])
                z = torch.randn_like(x_t) # Standard Gaussian noise
            else:
                sigma_t = torch.tensor(0.0, device=device) # Ensure on correct device
                z = torch.zeros_like(x_t) # Ensure on correct device

            # Reverse step (standard DDPM formula)
            term1_sqrt_alpha_t_inv = 1.0 / torch.sqrt(alpha_t)
            term2_coeff = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            x_t = term1_sqrt_alpha_t_inv * (x_t - term2_coeff * epsilon_theta) + sigma_t * z

        samples = x_t.cpu().numpy()
    return samples

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Added seconds for uniqueness

# ======================
# Evaluation and Visualization (Applicable to 1D Data)
# ======================
def inverse_transform_1d(samples, scaler):
    """Inverse Standardization for 1D Data"""
    N_samples, T_samples = samples.shape
    # Do not reshape samples to (-1, 1), but use original shape for inverse transformation
    samples_inv = scaler.inverse_transform(samples)
    return samples_inv

def evaluate_samples_1d(samples, scaler, time_stamp=None):
    """Evaluate and Visualize 1D Generated Samples"""
    print(">>> Step 6: Evaluation and Visualization (1D)")

    # 1. Inverse Standardization
    print("Inverse Standardizing the sample data...")
    samples_inversed = inverse_transform_1d(samples, scaler)
    save_dir = f"model_outputs_ddpm/eval_{time_stamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 2. Visualize a few samples
    print("Visualizing samples...")
    n_samples_to_plot = min(5, samples_inversed.shape[0]) # Plot at most 5 or available samples
    fig, axs = plt.subplots(n_samples_to_plot, 1, figsize=(12, 2.5 * n_samples_to_plot))
    if n_samples_to_plot == 1:
        axs = [axs]

    for i in range(n_samples_to_plot):
        axs[i].plot(samples_inversed[i, :])
        axs[i].set_title(f'Sample {i+1}')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"ddpm_samples_1d.png")
    plt.savefig(save_path)
    print(f"Samples image saved to {save_path}")
    plt.close(fig)

    # 3. Calculate covariance matrix
    print("Calculating covariance matrix...")
    Sigma_gen = np.cov(samples_inversed.T) # (T, T)

    # Plot covariance matrix
    fig_cov = plt.figure(figsize=(10, 8))
    plt.imshow(Sigma_gen, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Generated Samples Covariance Matrix')
    save_path_cov = os.path.join(save_dir, f"ddpm_covariance_1d.png")
    plt.savefig(save_path_cov)
    print(f"Covariance matrix image saved to {save_path_cov}")
    plt.close(fig_cov)

    print("Evaluation completed.")
    return samples_inversed

# ======================
# Save and Load Functions
# ======================
def save_outputs(samples, scaler, model_state_dict, time_stamp):
    """Save Generated Samples, Standardizer, and Model State"""
    save_dir = "model_outputs_ddpm"
    os.makedirs(save_dir, exist_ok=True)

    # Save samples
    np.save(os.path.join(save_dir, f"samples_ddpm_{time_stamp}.npy"), samples)
    print(f"Samples saved to {os.path.join(save_dir, f'samples_ddpm_{time_stamp}.npy')}")

    # Save standardizer
    joblib.dump(scaler, os.path.join(save_dir, f"scaler_ddpm_{time_stamp}.joblib"))
    print(f"Standardizer saved to {os.path.join(save_dir, f'scaler_ddpm_{time_stamp}.joblib')}")

    # Best model already saved through f"best_model_ddpm_{time_stamp}.pth" during training
    # If final model state needs to be saved, uncomment the following line
    # torch.save(model_state_dict, os.path.join(save_dir, f"final_model_ddpm_{time_stamp}.pth"))

def load_saved_outputs(time_stamp, config_T):
    """Load Saved Model, Samples, and Standardizer"""
    save_dir = "model_outputs_ddpm"
    model_path = f"best_model_ddpm_{time_stamp}.pth" # Load best model saved during training
    samples_path = os.path.join(save_dir, f"samples_ddpm_{time_stamp}.npy")
    scaler_path = os.path.join(save_dir, f"scaler_ddpm_{time_stamp}.joblib")

    loaded_model, loaded_samples, loaded_scaler = None, None, None

    if os.path.exists(model_path):
        loaded_model = DDPM(T_steps=config_T).to(device)
        loaded_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found.")

    if os.path.exists(samples_path):
        loaded_samples = np.load(samples_path)
        print(f"Samples loaded from {samples_path}")
    else:
        print(f"Error: Samples file {samples_path} not found.")
        
    if os.path.exists(scaler_path):
        loaded_scaler = joblib.load(scaler_path)
        print(f"Standardizer loaded from {scaler_path}")
    else:
        print(f"Error: Standardizer file {scaler_path} not found.")

    return loaded_model, loaded_samples, loaded_scaler

# ======================
# Main Flow
# ======================
def run_ddpm():
    # Initialize configuration
    config = AttrDict({
        'num_epochs': NUM_EPOCHS,
        'T': T, # Initial T value
        'device': device,
        'time_stamp': get_timestamp()
    })
    print(f">>> Step 0: Set Run Timestamp {config.time_stamp}")

    # Step 2: Data Loading (now return updated config)
    data_loader, scaler, test_size, config = load_data(config)
    if data_loader is None: 
        print("Data loading failed, cannot continue.")
        return

    # Step 1: Parameter Setup (after data loading, because T may be updated)
    config = setup_parameters(config)
    
    # Step 3: Model Setup
    model, optimizer, scheduler, loss_fn = setup_model(config)

    # Step 4: Train Model
    model = train_ddpm(model, data_loader, optimizer, scheduler, loss_fn, config)

    # Load best trained model for sampling
    best_model_path = f"best_model_ddpm_{config.time_stamp}.pth"
    if os.path.exists(best_model_path):
        print(f"Loading best model {best_model_path} for sampling...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model file not found, using last trained state model for sampling.")

    # Step 5: Sampling Generation
    samples_ddpm = inference(model, test_size, config)

    # Save output
    save_outputs(samples_ddpm, scaler, model.state_dict(), config.time_stamp)

    # Step 6: Evaluation and Visualization
    _ = evaluate_samples_1d(samples_ddpm, scaler, config.time_stamp)

    print(f"DDPM run completed, timestamp: {config.time_stamp}")

if __name__ == "__main__":
    run_ddpm()

