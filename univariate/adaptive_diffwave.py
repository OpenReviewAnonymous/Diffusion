# from matplotlib.dates import num2epoch
import numpy as np
from sympy import beta
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
    N, T, DATA_PATH, BATCH_SIZE
)

# LAMBDA_1 = 1e-1
# LAMBDA_1 = 1e-2
LAMBDA_1 = 5e-3
# LAMBDA_1 = 1e-3
LAMBDA_2 = 0
print(f"adaptive_diffwave: LAMBDA_1: {LAMBDA_1}, LAMBDA_2: {LAMBDA_2}")

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set random seed
np.random.seed(1010)
torch.manual_seed(1010)

# np.random.seed(42)
# torch.manual_seed(42)

# Time steps and sample size
# T = 100
# N = 100 * 1000
# DATA_PATH = "ar1_garch1_1_data.npy"


# CUDA_VISIBLE_DEVICES=5 python univariate/adaptive_diffwave.py 
LR_START = 1e-2
LR_END = 1e-5
# LR_END = 1e-6

NUM_EPOCHS = 500
# NUM_EPOCHS = 750
# NUM_EPOCHS = 1500
print(f"adaptive_diffwave: NUM_EPOCHS: {NUM_EPOCHS}, LR_START: {LR_START}, LR_END: {LR_END}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def silu(x):
    return x * torch.sigmoid(x)

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low)*(t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(64).unsqueeze(0)
        table = steps * 10.0**(dims * 4.0 / 63.0)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(512, residual_channels)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        y = self.dilated_conv(y)
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate)*torch.tanh(filt)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x+residual)/math.sqrt(2.0), skip

class AdaptiveDiffWaveModel(nn.Module):

    # ********* class DiffWaveModel(nn.Module): *********
    def __init__(self, T, residual_layers=30, residual_channels=64, dilation_cycle_length=10):
        super().__init__()
        self.T = T
        self.residual_channels = residual_channels
        self.diffusion_embedding = DiffusionEmbedding(T)
        self.input_projection = nn.Conv1d(1, residual_channels, 1)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels, 2**(i % dilation_cycle_length)) for i in range(residual_layers)
        ])
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
    # ********* class DiffWaveModel(nn.Module): *********
        
        # +++++++++++ Newly added for adaptive diffusion *********
        # Adaptive noise components
        self.L_params = nn.Parameter(torch.randn(T, T) * 0.01)
        self.register_buffer('tril_mask', torch.tril(torch.ones(T, T)))
        # +++++++++++ Newly added for adaptive diffusion *********

    # +++++++++++ Newly added for adaptive diffusion *********
    def get_L(self):
        L = self.L_params * self.tril_mask
        diag_indices = torch.arange(self.T)
        L[diag_indices, diag_indices] = F.softplus(L[diag_indices, diag_indices])
        return L
    # +++++++++++ Newly added for adaptive diffusion *********

    # +++++++++++ Newly added for adaptive diffusion *********
    def get_noise(self, batch_size):
        xi = torch.randn(batch_size, self.T, device=self.L_params.device)
        L = self.get_L()
        correlated_noise = xi @ L.T
        return correlated_noise
    # +++++++++++ Newly added for adaptive diffusion *********

    # ********* class DiffWaveModel(nn.Module): *********
    def forward(self, x, t):
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(t)
        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step)
            skip = skip_connection if skip is None else skip_connection + skip
        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x.squeeze(1) # (N,T)
    # ********* class DiffWaveModel(nn.Module): *********
    
    
    

logger = None  # Declare global logger variable

def setup_parameters(config):
    """Step 1: Set Parameters"""
    print(">>> Step 1: Set Parameters")
    # Setup noise schedule
    betas = torch.linspace(LR_START, 0.02, T, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    noise_schedule = torch.linspace(LR_START, 0.02, T, device=device)
    alphas_dw = 1 - noise_schedule
    alpha_bars_dw = torch.cumprod(alphas_dw, dim=0)
    # Add to config
    config.update({
        'noise_schedule': noise_schedule,
        'alphas_dw': alphas_dw,
        'alpha_bars_dw': alpha_bars_dw
    })
    
    return config

def load_data():
    """Step 2: Set Data"""
    print(">>> Step 2: Set Data")
    # DATA_PATH = "ar1_garch1_1_data_500.npy"
    
    volumes = np.load(DATA_PATH)
    print(f"DATA_PATH: {DATA_PATH}")
    # volumes = np.load("delta_data_electricity.npy")
    print(volumes.shape)
    # Number of time steps and samples

    # Calculate overall 99th percentile
    quantile = np.quantile(volumes, 0.99)

    # Find rows with values exceeding this percentile
    exceeding_mask = volumes > quantile

    # Find rows that have exceeding values
    rows_to_remove = np.any(exceeding_mask, axis=1)

    # Keep rows without exceeding values
    valid_rows = ~rows_to_remove
    volumes_filtered = volumes[valid_rows]

    print(f"Original shape: {volumes.shape}")
    volumes_filtered = volumes_filtered[:int(N*1.2),:T]
    print(f"Filtered shape: {volumes_filtered.shape}")

    prices = volumes_filtered.copy()
    S_true = prices

    # Split data into train (80%), validation (10%), and test (10%)
    total_samples = len(S_true)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    # Split the data
    train_data = S_true[:train_size]
    val_data = S_true[train_size:train_size+val_size]
    test_data = S_true[train_size+val_size:]

    # Data Normalization - fit only on training data
    scaler = StandardScaler()
    train_normalized = scaler.fit_transform(train_data)
    val_normalized = scaler.transform(val_data)
    test_normalized = scaler.transform(test_data)

    # Convert to tensors
    train_tensor = torch.tensor(train_normalized, dtype=torch.float32)
    val_tensor = torch.tensor(val_normalized, dtype=torch.float32)
    test_tensor = torch.tensor(test_normalized, dtype=torch.float32)

    # Create DataLoaders
    batch_size = BATCH_SIZE
    # batch_size = 64
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    test_dataset = TensorDataset(test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    data_loader = (train_loader, val_loader, test_loader)
    return data_loader, scaler, test_size


def setup_model(num_epochs):
    """Step 3: Set Model"""
    print(">>> Step 3: Set Model")
    model_adaptive = AdaptiveDiffWaveModel(T=T, residual_layers=30, residual_channels=64, 
                                         dilation_cycle_length=10).to(device)
    optimizer_adaptive = AdamW(model_adaptive.parameters(), lr=LR_START)
    scheduler_adaptive = CosineAnnealingLR(optimizer_adaptive, T_max=num_epochs, eta_min=LR_END)
    loss_fn_adaptive = nn.L1Loss()
    return model_adaptive, optimizer_adaptive, scheduler_adaptive, loss_fn_adaptive


# ************************
# **** Step 4 Start Training ****
# ************************
def train_adaptive_diffwave(model, optimizer, scheduler, data_loader, num_epochs, time_stamp, alpha_bars_dw, loss_fn_dw):
    print(">>> Step 4: Start Training")
    train_loader, val_loader, _ = data_loader
    best_val_loss = float('inf')
    model.train()
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/adaptive_diffwave_{time_stamp}')
    
    # Regularization coefficients
    lambda_1 = LAMBDA_1   # Frobenius norm regularization
    lambda_2 = LAMBDA_2  # Trace regularization
    
    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        train_loss = train_recon_loss = train_reg_loss = 0
        
        for batch in train_loader:
            x0 = batch[0].to(device)
            batch_size_ = x0.size(0)
            t = torch.randint(0, T, (batch_size_,), device=device)
            alpha_bar_t = alpha_bars_dw[t].view(-1,1)
            
            # Generate correlated noise using adaptive noise mechanism
            epsilon = model.get_noise(batch_size_)
            
            xt = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1 - alpha_bar_t)*epsilon
            epsilon_theta = model(xt, t)
            
            # Compute the residual
            r = epsilon_theta - epsilon
            
            # Get the Cholesky factor L
            L = model.get_L()
            
            # Solve L z = r^T for z using lower triangular solver
            r_T = r.T
            z_T = torch.linalg.solve_triangular(L, r_T, upper=False)
            z = z_T.T
            
            # Compute reconstruction loss
            recon_loss = torch.mean(torch.sum(z ** 2, dim=1))
            
            # Compute regularization losses
            frobenius_reg = torch.sum(L ** 2)
            trace_reg = torch.sum(torch.diagonal(L) ** 2)
            reg_scale = 20
            reg_loss = (lambda_1 * frobenius_reg + lambda_2 * trace_reg) * reg_scale
            
            # Total loss
            loss = recon_loss + reg_loss
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_reg_loss += reg_loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_loss = val_recon_loss = val_reg_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x0 = batch[0].to(device)
                batch_size_ = x0.size(0)
                t = torch.randint(0, T, (batch_size_,), device=device)
                alpha_bar_t = alpha_bars_dw[t].view(-1,1)
                
                epsilon = model.get_noise(batch_size_)
                xt = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1 - alpha_bar_t)*epsilon
                epsilon_theta = model(xt, t)
                
                r = epsilon_theta - epsilon
                L = model.get_L()
                r_T = r.T
                z_T = torch.linalg.solve_triangular(L, r_T, upper=False)
                z = z_T.T
                
                recon_loss = torch.mean(torch.sum(z ** 2, dim=1))
                frobenius_reg = torch.sum(L ** 2)
                reg_scale = 20
                trace_reg = torch.sum(torch.diagonal(L) ** 2) 
                reg_loss = (lambda_1 * frobenius_reg + lambda_2 * trace_reg) * reg_scale
                loss = recon_loss + reg_loss
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_reg_loss += reg_loss.item()
        
        scheduler.step()
        
        train_loss = train_loss/len(train_loader)
        train_recon_loss = train_recon_loss/len(train_loader)
        train_reg_loss = train_reg_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)
        val_recon_loss = val_recon_loss/len(val_loader)
        val_reg_loss = val_reg_loss/len(val_loader)
        
        # Log metrics to TensorBoard
        current_lr = scheduler.get_lr()[0]
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_recon', train_recon_loss, epoch)
        writer.add_scalar('Loss/train_reg', train_reg_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Loss/val_recon', val_recon_loss, epoch)
        writer.add_scalar('Loss/val_reg', val_reg_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_adaptive_diffwave_{time_stamp}.pth")
            
        print(f'[Adaptive DiffWave] Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, Reg: {train_reg_loss:.6f}), '
              f'Val Loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f}, Reg: {val_reg_loss:.6f}), '
              f'LR: {current_lr:.6f}')
    
    writer.close()
    return model


# **************************
# **** Step 5 Inference ****
# **************************
# ----------------- Sampling Comparison -----------------

def inference(model, N_test, T, alphas_dw, alpha_bars_dw, noise_schedule):
    # CHOICE A
    model.eval()
    print(">>> Step 5: Sampling")
    with torch.no_grad():
        x_t = torch.randn(N_test, T, device=device)
        for t_ in tqdm(reversed(range(T))):
            t_tensor = torch.full((N_test,), t_, dtype=torch.long, device=device)
            epsilon_theta = model(x_t, t_tensor)
            alpha_t = alphas_dw[t_]
            alpha_bar_t = alpha_bars_dw[t_]
            if t_ > 0:
                beta_t = noise_schedule[t_]
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
            else:
                sigma_t = 0
                z = 0
            x_t = (1/torch.sqrt(alpha_t))*(
                x_t - ((1 - alpha_t)/torch.sqrt(1 - alpha_bar_t))*epsilon_theta
            ) + sigma_t*z
        samples = x_t.cpu().numpy()
    return samples


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M:%S")

def run_adaptive_diffwave():
    # Step 0: Setup timestamp
    config = {
        'num_epochs': NUM_EPOCHS,
        'T': T,
        'device': device, 
        'time_stamp': get_timestamp()
    }
    print(f">>> Step 0: Setting up with timestamp {config['time_stamp']}")
    
    # Step 1: Setup parameters
    config = setup_parameters(config)
    
    # Step 2: Setup data
    data_loader, scaler, test_size = load_data()
    
    # Step 3: Setup model
    model_adaptive, optimizer_adaptive, scheduler_adaptive, loss_fn_adaptive =setup_model(config['num_epochs'])
    
    # Step 4: Training
    model_adaptive = train_adaptive_diffwave(model_adaptive, optimizer_adaptive, scheduler_adaptive, 
                                           data_loader, config['num_epochs'], config['time_stamp'],
                                           config['alpha_bars_dw'], loss_fn_adaptive)
    
    # Step 5: Sampling
    samples_adaptive = inference(model_adaptive, test_size, config['T'], 
                               config['alphas_dw'], config['alpha_bars_dw'], 
                               config['noise_schedule'])
    
    return data_loader, model_adaptive, samples_adaptive, scaler, config

if __name__ == "__main__":
    # Run the model
    data_loader, model_adaptive, samples_adaptive, scaler, config = run_adaptive_diffwave()

    # Save the samples and scaler
    save_dir = "model_outputs"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save samples
    np.save(f"{save_dir}/samples_adaptive_{config['time_stamp']}.npy", samples_adaptive)
    
    # Save scaler
    joblib.dump(scaler, f"{save_dir}/scaler_{config['time_stamp']}.joblib")
    
    print(f"Saved model outputs with timestamp: {config['time_stamp']}")
    
def load_saved_outputs(time_stamp):
    """Load saved model outputs for evaluation or further use.
    
    Args:
        time_stamp (str): Timestamp of the saved files in format 'YYYY-MM-DD_HH:MM'
        
    Returns:
        tuple: (model, samples, scaler)
    """
    save_dir = "model_outputs"
    
    # Load model
    model = AdaptiveDiffWaveModel(T=T, residual_layers=30, residual_channels=64, 
                                 dilation_cycle_length=10).to(device)
    model.load_state_dict(torch.load(f"best_model_adaptive_diffwave_univariate_{time_stamp}.pth"))
    
    # Load samples
    samples = np.load(f"{save_dir}/samples_adaptive_{time_stamp}.npy")
    
    # Load scaler
    scaler = joblib.load(f"{save_dir}/scaler_{time_stamp}.joblib")
    
    return model, samples, scaler



