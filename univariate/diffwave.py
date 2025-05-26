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
# If you need other schedulers such as ExponentialLR, you can add them, but here CosineAnnealingLR is used as an example.

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Number of time steps and samples
# T = 100
# N = 10000

T = 100

N = 100 * 1000

# DATA_PATH = "delta_data_electricity.npy"
DATA_PATH = "ar1_garch1_1_data.npy"

BATCH_SIZE = 10_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# ----------------- DiffWave DDPM Model Definition -----------------
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

class DiffWaveModel(nn.Module):
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
    
    
    
def load_data():
    """Step 2: Set up data"""
    print(">>> Step 2: Set up data")
    
    volumes = np.load(DATA_PATH)
    print(f"DATA_PATH: {DATA_PATH}")
    print(volumes.shape)
    # Number of time steps and samples


    # Calculate the overall 99% quantile
    quantile = np.quantile(volumes, 0.99)

    # Find out if each row has values exceeding this quantile
    exceeding_mask = volumes > quantile

    # Find rows with exceeding values
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
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    test_dataset = TensorDataset(test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    data_loader = (train_loader, val_loader, test_loader)
    return data_loader, scaler, test_size

def setup_parameters(config):
    """Step 1: Set up parameters"""
    print(">>> Step 1: Set up parameters")
    # Setup noise schedule
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    noise_schedule = torch.linspace(1e-4, 0.02, T, device=device)
    alphas_dw = 1 - noise_schedule
    alpha_bars_dw = torch.cumprod(alphas_dw, dim=0)
    
    # Add to config
    config.update({
        'noise_schedule': noise_schedule,
        'alphas_dw': alphas_dw,
        'alpha_bars_dw': alpha_bars_dw
    })
    
    return config

def setup_data():
    """Step 2: Set up data"""
    print(">>> Step 2: Set up data")
    return load_data()

def setup_model(num_epochs):
    """Step 3: Set up model"""
    print(">>> Step 3: Set up model")
    model_diffwave = DiffWaveModel(T=T, residual_layers=30, residual_channels=64, 
                                 dilation_cycle_length=10).to(device)
    
    diffwave_lr = 1e-4
    optimizer_diffwave = AdamW(model_diffwave.parameters(), lr=diffwave_lr)
    loss_fn_dw = nn.L1Loss()
    
    scheduler_diffwave = CosineAnnealingLR(optimizer_diffwave, T_max=num_epochs, eta_min=1e-6)
    
    return model_diffwave, optimizer_diffwave, scheduler_diffwave, loss_fn_dw


# ************************
# **** step 4 start training ****
# ************************
def train_diffwave(model_diffwave, optimizer_diffwave, scheduler_diffwave, 
                  data_loader, num_epochs, time_stamp,
                  alpha_bars_dw, loss_fn_dw):
    print(">>> Step 4: Start training")
    train_loader, val_loader, _ = data_loader
    best_val_loss = float('inf')
    model_diffwave.train()
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/diffwave_{time_stamp}')
    
    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model_diffwave.train()
        train_loss = 0
        for batch in train_loader:
            x0 = batch[0].to(device)
            batch_size_ = x0.size(0)
            t = torch.randint(0, T, (batch_size_,), device=device)
            alpha_bar_t = alpha_bars_dw[t].view(-1,1)
            epsilon = torch.randn_like(x0)
            xt = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1 - alpha_bar_t)*epsilon
            epsilon_theta = model_diffwave(xt, t)
            loss = loss_fn_dw(epsilon_theta, epsilon)
            train_loss += loss.item()
            optimizer_diffwave.zero_grad()
            loss.backward()
            optimizer_diffwave.step()
        
        # Validation phase
        model_diffwave.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x0 = batch[0].to(device)
                batch_size_ = x0.size(0)
                t = torch.randint(0, T, (batch_size_,), device=device)
                alpha_bar_t = alpha_bars_dw[t].view(-1,1)
                epsilon = torch.randn_like(x0)
                xt = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1 - alpha_bar_t)*epsilon
                epsilon_theta = model_diffwave(xt, t)
                loss = loss_fn_dw(epsilon_theta, epsilon)
                val_loss += loss.item()
        
        scheduler_diffwave.step()
        
        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)
        
        # ***** Log metrics to TensorBoard ****
        current_lr = scheduler_diffwave.get_lr()[0]
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        # ***** Log metrics to TensorBoard ****
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model here if needed
            torch.save(model_diffwave.state_dict(), f"best_model_diffwave_{time_stamp}.pth")
        print(f'[DiffWave DDPM] Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')
    
    writer.close()
    return model_diffwave

# **************************
# **** step 5 inference ****
# **************************
# ----------------- Sampling comparison -----------------
def inference(model_diffwave, N_test, T, alphas_dw, alpha_bars_dw, noise_schedule):
    model_diffwave.eval()
    print(">>> Step 5: Sampling comparison")
    with torch.no_grad():
        x_t = torch.randn(N_test, T, device=device)
        for t_ in tqdm(reversed(range(T))):
            t_tensor = torch.full((N_test,), t_, dtype=torch.long, device=device)
            epsilon_theta = model_diffwave(x_t, t_tensor)
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
        samples_diffwave = x_t.cpu().numpy()
    return samples_diffwave

def get_timestamp():
    """Generate a timestamp in the format YYYY-MM-DD_HH"""
    return datetime.now().strftime("%Y-%m-%d_%H:%M")

def run_diffwave():
    # Step 0: Setup timestamp
    config = {
        'num_epochs': 500,
        'T': T,
        'device': device, 
        'time_stamp': get_timestamp()
    }
    print(f">>> Step 0: Setting up with timestamp {config['time_stamp']}")
    
    # Step 1: Set up parameters
    config = setup_parameters(config)
    
    # Step 2: Set up data
    data_loader, scaler, test_size = setup_data()
    
    # Step 3: Set up model
    model_diffwave, optimizer_diffwave, scheduler_diffwave, loss_fn_dw = setup_model(config['num_epochs'])
    
    # Step 4: Start training
    model_diffwave = train_diffwave(model_diffwave, optimizer_diffwave, scheduler_diffwave, 
                                  data_loader, config['num_epochs'], config['time_stamp'] , config['alpha_bars_dw'],
                                  loss_fn_dw) 
    
    # Step 5: Sampling comparison - use test_size for inference
    samples_diffwave = inference(model_diffwave, test_size, config['T'], 
                               config['alphas_dw'], config['alpha_bars_dw'], 
                               config['noise_schedule']) 

    return data_loader, model_diffwave, samples_diffwave, scaler

if __name__ == "__main__":
    data_loader, model_diffwave, samples_diffwave, scaler = run_diffwave()