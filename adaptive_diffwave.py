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
from diffwave import load_data, BATCH_SIZE
import ot  # For Wasserstein distance (pip install pot)
import logging
import time

logger = None  # 声明全局logger变量

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Time steps and sample size
T = 100
N = 100 * 1000
# 从环境变量或使用默认值读取超参数
LAMBDA_RECON = float(os.environ.get("HP_LAMBDA_RECON", "1")) 
LAMBDA_REG = float(os.environ.get("HP_LAMBDA_REG", "50")) 
# LAMBDA_REG = float(os.environ.get("HP_LAMBDA_REG", "1")) 
DIFFWAVE_LR = float(os.environ.get("HP_DIFFWAVE_LR", "1e-3"))
DIFFWAVE_LR_ETAMIN = float(os.environ.get("HP_DIFFWAVE_LR_ETAMIN", "1e-5"))
# DIFFWAVE_LR_ETAMIN = float(os.environ.get("HP_DIFFWAVE_LR_ETAMIN", "1e-6"))
# BATCH_SIZE = int(os.environ.get("HP_BATCH_SIZE", "10_000"))
# BATCH_SIZE = int(os.environ.get("HP_BATCH_SIZE", "1024"))
# BATCH_SIZE = int(os.environ.get("HP_BATCH_SIZE", "64"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.input_projection = nn.Conv1d(2, residual_channels, 1)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels, 2**(i % dilation_cycle_length)) for i in range(residual_layers)
        ])
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2, 1)
        nn.init.zeros_(self.output_projection.weight)
    # ********* class DiffWaveModel(nn.Module): *********
        
        # +++++++++++ Newly added for adaptive diffusion *********
        # Adaptive noise components
        self.T_total = 2 * T
        self.L_params = nn.Parameter(torch.randn(self.T_total, self.T_total) * 0.01)
        self.register_buffer('tril_mask', torch.tril(torch.ones(self.T_total, self.T_total)))
        # +++++++++++ Newly added for adaptive diffusion *********

    # +++++++++++ Newly added for adaptive diffusion *********
    def get_L(self):
        L = self.L_params * self.tril_mask
        diag_indices = torch.arange(self.T_total)
        L[diag_indices, diag_indices] = F.softplus(L[diag_indices, diag_indices])
        return L
    # +++++++++++ Newly added for adaptive diffusion *********

    # +++++++++++ Newly added for adaptive diffusion *********
    def get_noise(self, batch_size):
        xi = torch.randn(batch_size, self.T_total, device=self.L_params.device)
        L = self.get_L()
        correlated_noise = xi @ L.T
        correlated_noise = correlated_noise.reshape(batch_size, self.T, 2)
        return correlated_noise
    # +++++++++++ Newly added for adaptive diffusion *********

    # ********* class DiffWaveModel(nn.Module): *********
    def forward(self, x, t):
        x = x.permute(0, 2, 1)
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
        x = x.permute(0, 2, 1)
        return x
    # ********* class DiffWaveModel(nn.Module): *********
    
    
    

def setup_parameters(config):
    """Step 1: Setup parameters"""
    global logger
    logger.info(">>> Step 1: Setup parameters")
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


def setup_model(num_epochs):
    """Step 3: Setup model"""
    global logger
    logger.info(">>> Step 3: Setup model")
    model_adaptive = AdaptiveDiffWaveModel(T=T, residual_layers=30, residual_channels=64, 
                                         dilation_cycle_length=10).to(device)
    optimizer_adaptive = AdamW(model_adaptive.parameters(), lr=DIFFWAVE_LR)
    scheduler_adaptive = CosineAnnealingLR(optimizer_adaptive, T_max=num_epochs, eta_min=DIFFWAVE_LR_ETAMIN)
    loss_fn_adaptive = nn.L1Loss()
    return model_adaptive, optimizer_adaptive, scheduler_adaptive, loss_fn_adaptive


# ************************
# **** step 4 Start training ****
# ************************
def train_adaptive_diffwave(model, optimizer, scheduler, data_loader, num_epochs, config, loss_fn_dw):
    global logger
    logger.info(">>> Step 4: Start training")
    train_loader, val_loader, _ = data_loader
    best_val_loss = float('inf')
    model.train()
    
    # Read early stopping parameters from environment variables
    patience = int(os.environ.get("HP_PATIENCE", "500"))
    min_delta = float(os.environ.get("HP_MIN_DELTA", "1e-5"))
    logger.info(f"Early stopping patience: {patience}, min delta: {min_delta}")
    
    # Early stopping related variables
    epochs_no_improve = 0
    early_stop = False
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/adaptive_diffwave_{config["run_id"]}')
    
    # Read regularization coefficients from environment variables
    lambda_1 = float(os.environ.get("HP_LAMBDA_1", "1e-3"))  # Frobenius regularization
    lambda_2 = float(os.environ.get("HP_LAMBDA_2", "0"))  # Trace regularization
    
    # Pre-compute parameters
    alpha_bars_dw = config['alpha_bars_dw']
    alphas_dw = config['alphas_dw']
    noise_schedule = config['noise_schedule']
    
    # Variables for recording time
    time_total = 0
    iter_count = 0
    
    for epoch in tqdm(range(num_epochs)):
        if early_stop:
            logger.info("Early stopping triggered.")
            break
        
        # Training phase
        model.train()
        train_loss = train_recon_loss = train_reg_loss = 0
        train_weighted_reg_loss = 0
        train_weighted_recon_loss = 0
        
        for batch in train_loader:
            iter_start_time = time.time()
            iter_count += 1
            
            # Data preparation
            x0 = batch[0].to(device)
            batch_size_ = x0.size(0)
            t = torch.randint(0, T, (batch_size_,), device=device)
            alpha_bar_t = alpha_bars_dw[t].view(-1, 1, 1)
            
            # Generate correlated noise using adaptive noise mechanism
            epsilon = model.get_noise(batch_size_)
            
            xt = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1 - alpha_bar_t)*epsilon
            
            # Forward propagation
            epsilon_theta = model(xt, t)
            
            # Calculate loss
            # Compute the residual
            r = (epsilon_theta - epsilon).reshape(batch_size_, -1)
            
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
            reg_loss = lambda_1 * frobenius_reg + lambda_2 * trace_reg
            
            # Total loss
            lambda_recon = LAMBDA_RECON
            lambda_reg = LAMBDA_REG
            weighted_reg_loss = lambda_reg * reg_loss
            weighted_recon_loss = lambda_recon * recon_loss
            loss = weighted_recon_loss + weighted_reg_loss
            
            # Record weighted regularization loss (record every 100 iterations)
            # if iter_count % 100 == 0:
            logger.info(f"Train iter {iter_count}: lambda_reg * reg_loss = {weighted_reg_loss.item():.6f}, lambda_recon * recon_loss = {weighted_recon_loss.item():.6f}")
            
            # Record both weighted regularization loss and reconstruction loss
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_reg_loss += reg_loss.item()
            train_weighted_reg_loss += weighted_reg_loss.item()
            train_weighted_recon_loss += weighted_recon_loss.item()
            
            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Record total time
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            time_total += iter_time
            
            # Record iteration time
            if iter_count % 100 == 0:
                avg_time = time_total / 100
                logger.info(f"Iteration {iter_count}, avg time per iter: {avg_time:.4f} seconds")
                time_total = 0
        
        # Validation phase
        model.eval()
        val_loss = val_recon_loss = val_reg_loss = 0
        val_weighted_reg_loss = 0
        val_weighted_recon_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x0 = batch[0].to(device)
                batch_size_ = x0.size(0)
                t = torch.randint(0, T, (batch_size_,), device=device)
                alpha_bar_t = alpha_bars_dw[t].view(-1, 1, 1)
                
                epsilon = model.get_noise(batch_size_)
                xt = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1 - alpha_bar_t)*epsilon
                epsilon_theta = model(xt, t)
                
                r = (epsilon_theta - epsilon).reshape(batch_size_, -1)
                
                L = model.get_L()
                r_T = r.T
                z_T = torch.linalg.solve_triangular(L, r_T, upper=False)
                z = z_T.T
                
                recon_loss = torch.mean(torch.sum(z ** 2, dim=1))
                frobenius_reg = torch.sum(L ** 2)
                trace_reg = torch.sum(torch.diagonal(L) ** 2)
                reg_loss = lambda_1 * frobenius_reg + lambda_2 * trace_reg
                
                # Total loss
                lambda_recon = LAMBDA_RECON
                lambda_reg = LAMBDA_REG
                weighted_reg_loss = lambda_reg * reg_loss
                weighted_recon_loss = lambda_recon * recon_loss
                loss = weighted_recon_loss + weighted_reg_loss
                
                # Record weighted regularization loss
                # if batch_idx == 0 and epoch % 10 == 0:  # Record first validation batch every 10 epochs
                logger.info(f"Val epoch {epoch+1}, batch {batch_idx}: lambda_reg * reg_loss = {weighted_reg_loss.item():.6f}, lambda_recon * recon_loss = {weighted_recon_loss.item():.6f}")
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_reg_loss += reg_loss.item()
                val_weighted_reg_loss += weighted_reg_loss.item()
                val_weighted_recon_loss += weighted_recon_loss.item()
        
        avg_train_loss = train_loss/len(train_loader)
        avg_train_recon_loss = train_recon_loss/len(train_loader)
        avg_train_reg_loss = train_reg_loss/len(train_loader)
        avg_train_weighted_reg_loss = train_weighted_reg_loss/len(train_loader)
        avg_train_weighted_recon_loss = train_weighted_recon_loss/len(train_loader)
        
        avg_val_loss = val_loss/len(val_loader)
        avg_val_recon_loss = val_recon_loss/len(val_loader)
        avg_val_reg_loss = val_reg_loss/len(val_loader)
        avg_val_weighted_reg_loss = val_weighted_reg_loss/len(val_loader)
        avg_val_weighted_recon_loss = val_weighted_recon_loss/len(val_loader)
        
        # Record current LAMBDA_RECON and LAMBDA_REG values
        logger.info(f"Current hyperparameters - LAMBDA_RECON: {LAMBDA_RECON}, LAMBDA_REG: {LAMBDA_REG}")
        logger.info(f"Debug - train_recon: {avg_train_recon_loss:.4f} * {LAMBDA_RECON} = {avg_train_recon_loss * LAMBDA_RECON:.4f}")
        logger.info(f"Debug - train_reg: {avg_train_reg_loss:.4f} * {LAMBDA_REG} = {avg_train_reg_loss * LAMBDA_REG:.4f}")
        logger.info(f"Debug - val_recon: {avg_val_recon_loss:.4f} * {LAMBDA_RECON} = {avg_val_recon_loss * LAMBDA_RECON:.4f}")
        logger.info(f"Debug - val_reg: {avg_val_reg_loss:.4f} * {LAMBDA_REG} = {avg_val_reg_loss * LAMBDA_REG:.4f}")
        
        scheduler.step()
        
        # Log metrics to TensorBoard
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/train: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train_recon', avg_train_recon_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/train_recon: {avg_train_recon_loss:.4f}")
        writer.add_scalar('Loss/train_reg', avg_train_reg_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/train_reg: {avg_train_reg_loss:.4f}")
        writer.add_scalar('Loss/train_weighted_reg', avg_train_weighted_reg_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/train_weighted_reg: {avg_train_weighted_reg_loss:.4f}")
        writer.add_scalar('Loss/train_weighted_recon', avg_train_weighted_recon_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/train_weighted_recon: {avg_train_weighted_recon_loss:.4f}")
        
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/val: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/val_recon', avg_val_recon_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/val_recon: {avg_val_recon_loss:.4f}")
        writer.add_scalar('Loss/val_reg', avg_val_reg_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/val_reg: {avg_val_reg_loss:.4f}")
        writer.add_scalar('Loss/val_weighted_reg', avg_val_weighted_reg_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/val_weighted_reg: {avg_val_weighted_reg_loss:.4f}")
        writer.add_scalar('Loss/val_weighted_recon', avg_val_weighted_recon_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/val_weighted_recon: {avg_val_weighted_recon_loss:.4f}")
        
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Learning_Rate: {current_lr:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), f"best_model_adaptive_diffwave_{config['run_id']}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
        

        
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon_loss:.4f}, Reg: {avg_train_reg_loss:.4f}) | Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon_loss:.4f}, Reg: {avg_val_reg_loss:.4f}) | LR: {current_lr:.2e} ")
    
    writer.close()
    return model


# **************************
# **** step 5 inference ****
# **************************
# ----------------- Sampling comparison -----------------

def inference(model, N_test, T, alphas_dw, alpha_bars_dw, noise_schedule):
    global logger
    model.eval()
    logger.info(">>> Step 5: Sampling")
    with torch.no_grad():
        x_t = torch.randn(N_test, T, 2, device=device)
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
    return datetime.now().strftime("%Y%m%d_%H%M-%S")

def run_adaptive_diffwave():
    # Initialize configuration
    run_id = os.environ.get("RUN_ID", get_timestamp())  # Get from environment variables or generate run_id
    num_epochs = int(os.environ.get("HP_NUM_EPOCHS", "750"))  # Read epochs from environment variables
    
    # Setup logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"adaptive_diffwave_{run_id}.log")
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s',
                      handlers=[
                          logging.FileHandler(log_file_path),
                          logging.StreamHandler()  # Also output to console
                      ])
    global logger  # Declare modification of global logger variable
    logger = logging.getLogger(__name__)  # Initialize global logger
    
    # Read early stopping parameters from environment variables
    patience = int(os.environ.get("HP_PATIENCE", "100"))
    min_delta = float(os.environ.get("HP_MIN_DELTA", "1e-5"))
    
    # Record initial hyperparameters and device information
    logger.info(f"LAMBDA_RECON: {LAMBDA_RECON}, LAMBDA_REG: {LAMBDA_REG}")
    logger.info(f"DIFFWAVE_LR: {DIFFWAVE_LR}")
    logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"Using device: {device}")
    
    config = {
        # 'num_epochs': 750,
        'num_epochs': 1250,
        # 'num_epochs': num_epochs,
        'T': T,
        'device': device, 
        'run_id': run_id,
        'patience': patience,
        'min_delta': min_delta
    }
    
    # Record initial run parameters
    logger.info(f"Starting run with ID: {run_id}")
    logger.info(f"HP_NUM_EPOCHS: {num_epochs}")
    logger.info(f"HP_LAMBDA_1: {os.environ.get('HP_LAMBDA_1', '1e-3')}")
    logger.info(f"HP_LAMBDA_2: {os.environ.get('HP_LAMBDA_2', '0')}")
    logger.info(f"HP_PATIENCE: {patience}")
    logger.info(f"HP_MIN_DELTA: {min_delta}")
    
    # Step 1: Setup parameters
    config = setup_parameters(config)
    
    # Step 2: Setup data
    logger.info(">>> Step 2: Data loading")
    data_loader, scalers, test_size = load_data()
    
    # Step 3: Setup model
    model_adaptive, optimizer_adaptive, scheduler_adaptive, loss_fn_adaptive = setup_model(config['num_epochs'])
    
    # Step 4: Training
    model_adaptive = train_adaptive_diffwave(model_adaptive, optimizer_adaptive, scheduler_adaptive, 
                                           data_loader, config['num_epochs'], config,
                                           loss_fn_adaptive)
    
    # After early stopping, we should load the saved best model weights for inference
    best_model_path = f"best_model_adaptive_diffwave_{config['run_id']}.pth"
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        model_adaptive.load_state_dict(torch.load(best_model_path))
    else:
        logger.warning(f"Warning: Best model not found at {best_model_path}. Using the model from the last epoch.")
    
    # Step 5: Sampling
    samples_adaptive = inference(model_adaptive, test_size, config['T'], 
                               config['alphas_dw'], config['alpha_bars_dw'], 
                               config['noise_schedule'])
    
    # Save results
    save_dir = f"results_diffwave_{config['run_id']}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/samples.npy", samples_adaptive)
    joblib.dump(scalers, f"{save_dir}/scalers.joblib")
    logger.info(f"Results saved to {save_dir}")
    # Print generated sample path for evaluation script use
    logger.info(f"Generated samples path: {os.path.abspath(f'{save_dir}/samples.npy')}")
    
    return data_loader, model_adaptive, samples_adaptive, scalers, config

if __name__ == "__main__":
    # Run the model
    data_loader, model_adaptive, samples_adaptive, scalers, config = run_adaptive_diffwave()
    
def load_saved_outputs(run_id):
    """Load saved model outputs for evaluation or further use.
    
    Args:
        run_id (str): Run ID of the saved files
        
    Returns:
        tuple: (model, samples, scaler)
    """
    save_dir = f"results_diffwave_{run_id}"
    
    # Load model
    model = AdaptiveDiffWaveModel(T=T, residual_layers=30, residual_channels=64, 
                                 dilation_cycle_length=10).to(device)
    model.load_state_dict(torch.load(f"best_model_adaptive_diffwave_{run_id}.pth"))
    
    # Load samples
    samples = np.load(f"{save_dir}/samples.npy")
    
    # Load scaler
    scaler = joblib.load(f"{save_dir}/scalers.joblib")
    
    return model, samples, scaler



