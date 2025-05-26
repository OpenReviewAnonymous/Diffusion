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
# If other schedulers like ExponentialLR are needed, they can be added, but here we use CosineAnnealingLR as an example.

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Number of time steps and samples
# T = 100
# N = 10000

T = 100

N = 100 * 1000
# DIFFWAVE_LR = 1e-3
# DIFFWAVE_LR = 1e-4
DIFFWAVE_LR = 1e-6
DIFFWAVE_LR_ETAMIN = 1e-7
# DIFFWAVE_LR = 5e-5
print(f"DIFFWAVE_LR: {DIFFWAVE_LR}")

# BATCH_SIZE = 64
# BATCH_SIZE = 2048
# BATCH_SIZE = 4096
BATCH_SIZE = 10_000
print(f"BATCH_SIZE: {BATCH_SIZE}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



    
    
# ----------------- DiffWave Model Definition -----------------
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
    """
    Modified to handle 2D input:
      - input_channels = 2 instead of 1
      - output_channels = 2 instead of 1
    """
    def __init__(self, T, residual_layers=30, residual_channels=64, dilation_cycle_length=10):
        super().__init__()
        self.T = T
        self.residual_channels = residual_channels
        self.diffusion_embedding = DiffusionEmbedding(T)
        
        # Changed in_channels=2 for 2D time series
        self.input_projection = nn.Conv1d(in_channels=2, out_channels=residual_channels, kernel_size=1)
        
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels, 2**(i % dilation_cycle_length)) for i in range(residual_layers)
        ])
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 1)
        
        # Changed out_channels=2 for 2D time series
        self.output_projection = nn.Conv1d(residual_channels, 2, 1)
        
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, t):
        """
        x shape: (batch_size, T, 2)
        We'll permute to (batch_size, 2, T) for Conv1d, then permute back.
        """
        # x: (N, T, 2) -> (N, 2, T)
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
        
        # (N, 2, T) -> (N, T, 2)
        x = x.permute(0, 2, 1)
        return x
    
def setup_parameters(config):
    """Step 1: Set parameters"""
    print(">>> Step 1: Set parameters")
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
    

def load_data():
    """Step 2: Prepare data"""
    print(">>> Step 2: Prepare data")
    volumes = np.load("./two_dimension_garch.npy")
    print("Loaded 2D data with shape:", volumes.shape)
    
    N_all, T_data, D_data = volumes.shape
    volumes_filtered = volumes.copy()

    print(f"Original shape: {volumes.shape}")
    volumes_filtered = volumes_filtered[:int(N*1.2), :T, :]
    print(f"Filtered shape: {volumes_filtered.shape}")
    
    S_true = volumes_filtered

    # Split data into train (80%), validation (10%), and test (10%)
    total_samples = len(S_true)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    # Split the data
    train_data = S_true[:train_size]
    val_data = S_true[train_size:train_size+val_size]
    test_data = S_true[train_size+val_size:]

    # ---------------------------------- #H1
    # 1) Extract X and Y channels separately (train, val, test)
    train_X = train_data[..., 0]  # shape (train_size, T)
    train_Y = train_data[..., 1]  # shape (train_size, T)
    val_X   = val_data[..., 0]    # shape (val_size, T)
    val_Y   = val_data[..., 1]
    test_X  = test_data[..., 0]   # shape (test_size, T)
    test_Y  = test_data[..., 1]

    # 2) 分别展开成 (N*T, 1)，以便对每个通道单独做标准化
    train_X_flat = train_X.reshape(-1, 1)
    train_Y_flat = train_Y.reshape(-1, 1)
    val_X_flat   = val_X.reshape(-1, 1)
    val_Y_flat   = val_Y.reshape(-1, 1)
    test_X_flat  = test_X.reshape(-1, 1)
    test_Y_flat  = test_Y.reshape(-1, 1)

    # 3) 为 X、Y 分别创建 scaler
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # 只在训练集上 fit
    train_X_scaled_flat = scaler_x.fit_transform(train_X_flat)
    train_Y_scaled_flat = scaler_y.fit_transform(train_Y_flat)

    # 验证集与测试集只 transform
    val_X_scaled_flat  = scaler_x.transform(val_X_flat)
    val_Y_scaled_flat  = scaler_y.transform(val_Y_flat)
    test_X_scaled_flat = scaler_x.transform(test_X_flat)
    test_Y_scaled_flat = scaler_y.transform(test_Y_flat)

    # 4) 再 reshape 回 (N, T)
    train_X_scaled = train_X_scaled_flat.reshape(train_size, T)
    train_Y_scaled = train_Y_scaled_flat.reshape(train_size, T)
    val_X_scaled   = val_X_scaled_flat.reshape(val_size, T)
    val_Y_scaled   = val_Y_scaled_flat.reshape(val_size, T)
    test_X_scaled  = test_X_scaled_flat.reshape(test_size, T)
    test_Y_scaled  = test_Y_scaled_flat.reshape(test_size, T)

    # 5) 最后合并回 (N, T, 2)
    train_scaled = np.stack([train_X_scaled, train_Y_scaled], axis=-1)  # (train_size, T, 2)
    val_scaled   = np.stack([val_X_scaled,   val_Y_scaled],   axis=-1)  # (val_size, T, 2)
    test_scaled  = np.stack([test_X_scaled,  test_Y_scaled],  axis=-1)  # (test_size, T, 2)

    # 把两个 scaler 合并打包，以便后续保存
    # 也可用元组 (scaler_x, scaler_y) 或字典等方式
    scalers = {
        'x': scaler_x,
        'y': scaler_y
    }
    # ---------------------------------- #H1

    # 转成张量
    train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
    val_tensor = torch.tensor(val_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

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
    return data_loader, scalers, test_size



def setup_model(num_epochs):
    """Step 3: 设置模型"""
    print(">>> Step 3: 设置模型")
    model_diffwave = DiffWaveModel(T=T, residual_layers=30, residual_channels=64, 
                                 dilation_cycle_length=10).to(device)
    
    diffwave_lr = DIFFWAVE_LR
    optimizer_diffwave = AdamW(model_diffwave.parameters(), lr=diffwave_lr)
    loss_fn_dw = nn.MSELoss()
    # loss_fn_dw = nn.L1Loss()
    print(f"loss_fn_dw: {loss_fn_dw}")
    
    scheduler_diffwave = CosineAnnealingLR(optimizer_diffwave, T_max=num_epochs, eta_min=DIFFWAVE_LR_ETAMIN)
    
    return model_diffwave, optimizer_diffwave, scheduler_diffwave, loss_fn_dw


# ************************
# **** step 4 开始训练 ****
# ************************
def train_diffwave(model_diffwave, optimizer_diffwave, scheduler_diffwave, 
                  data_loader, num_epochs, time_stamp,
                  alpha_bars_dw, loss_fn_dw):
    print(">>> Step 4: 开始训练")
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
            x0 = batch[0].to(device)  # shape: (batch_size, T, 2)
            batch_size_ = x0.size(0)
            t = torch.randint(0, T, (batch_size_,), device=device)
            alpha_bar_t = alpha_bars_dw[t].view(-1, 1, 1) # 修改alpha_bar_t的形状以便广播到(batch_size, T, 2)
            epsilon = torch.randn_like(x0)  # shape: (batch_size, T, 2)
            xt = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1 - alpha_bar_t)*epsilon
            epsilon_theta = model_diffwave(xt, t)  # shape: (batch_size, T, 2)
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
                x0 = batch[0].to(device)  # shape: (batch_size, T, 2)
                batch_size_ = x0.size(0)
                t = torch.randint(0, T, (batch_size_,), device=device)
                alpha_bar_t = alpha_bars_dw[t].view(-1, 1, 1) # 修改alpha_bar_t的形状以便广播到(batch_size, T, 2)
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
# ----------------- 采样对比 -----------------
def inference(model_diffwave, N_test, T, alphas_dw, alpha_bars_dw, noise_schedule):
    model_diffwave.eval()
    print(">>> Step 5: 采样对比")
    with torch.no_grad():
        x_t = torch.randn(N_test, T, 2, device=device) # 初始化具有(N_test, T, 2)形状的噪声
        for t_ in tqdm(reversed(range(T))):
            t_tensor = torch.full((N_test,), t_, dtype=torch.long, device=device)
            epsilon_theta = model_diffwave(x_t, t_tensor)  # shape: (N_test, T, 2)
            alpha_t = alphas_dw[t_]
            alpha_bar_t = alpha_bars_dw[t_]
            if t_ > 0:
                beta_t = noise_schedule[t_]
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
            else:
                sigma_t = 0
                z = 0
                
            # 标量alpha_t可以广播到x_t的形状(N_test, T, 2)
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
        'num_epochs': 200,
        'T': T,
        'device': device, 
        'time_stamp': get_timestamp()
    }
    print(f">>> Step 0: Setting up with timestamp {config['time_stamp']}")
    
    # Step 1: 设置参数
    config = setup_parameters(config)
    
    # Step 2: 设置数据
    data_loader, scalers, test_size = load_data()
    
    # Step 3: 设置模型
    model_diffwave, optimizer_diffwave, scheduler_diffwave, loss_fn_dw = setup_model(config['num_epochs'])
    
    # Step 4: 开始训练
    model_diffwave = train_diffwave(model_diffwave, optimizer_diffwave, scheduler_diffwave, 
                                  data_loader, config['num_epochs'], config['time_stamp'] , config['alpha_bars_dw'],
                                  loss_fn_dw) 
    
    # Step 5: 采样对比 - use test_size for inference
    samples_diffwave = inference(model_diffwave, test_size, config['T'], 
                               config['alphas_dw'], config['alpha_bars_dw'], 
                               config['noise_schedule']) 

    return data_loader, model_diffwave, samples_diffwave, scalers, config['time_stamp']

if __name__ == "__main__":

    # 运行模型
    data_loader, model_diffwave, samples_diffwave, scalers, time_stamp = run_diffwave()
    
    # 创建保存目录
    save_dir = "model_outputs"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存样本
    np.save(f"{save_dir}/samples_diffwave_{time_stamp}.npy", samples_diffwave)
    
    # 保存标准化器
    joblib.dump(scalers, f"{save_dir}/scalers_{time_stamp}.joblib")
    
    
    # # 评估和可视化生成的样本
    # print("开始评估和可视化生成的样本...")
    # evaluate_samples(samples_diffwave, scalers, time_stamp)

    
    print(f"已保存模型输出，时间戳：{time_stamp}")

def load_saved_outputs(time_stamp):
    """加载已保存的模型输出，用于评估或进一步使用。
    
    参数:
        time_stamp (str): 保存文件的时间戳，格式为'YYYY-MM-DD_HH:MM'
        
    返回:
        tuple: (model, samples, scalers)
    """
    save_dir = "model_outputs"
    
    # 加载模型
    model = DiffWaveModel(T=T, residual_layers=30, residual_channels=64, 
                          dilation_cycle_length=10).to(device)
    model.load_state_dict(torch.load(f"best_model_diffwave_{time_stamp}.pth"))
    
    # 加载样本
    samples = np.load(f"{save_dir}/samples_diffwave_{time_stamp}.npy")
    
    # 加载标准化器
    scalers = joblib.load(f"{save_dir}/scalers_{time_stamp}.joblib")
    
    return model, samples, scalers

# ---------------------------------- #H1
def inverse_transform_channels(samples, scalers):
    """
    对两个通道分别使用各自的scaler进行逆标准化。
    
    参数:
        samples: 形状为 (N, T, 2) 的样本数据
        scalers: 包含 'x' 和 'y' 两个 scaler 的字典
        
    返回:
        inverse_samples: 逆标准化后的样本数据，形状为 (N, T, 2)
    """
    n_samples = samples.shape[0]
    T_samples = samples.shape[1]
    
    # 1) 分离两个通道
    channel_x = samples[..., 0]  # (N, T)
    channel_y = samples[..., 1]  # (N, T)
    
    # 2) 重塑为 (N*T, 1) 以便应用 inverse_transform
    channel_x_flat = channel_x.reshape(-1, 1)
    channel_y_flat = channel_y.reshape(-1, 1)
    
    # 3) 分别应用逆标准化
    channel_x_inv_flat = scalers['x'].inverse_transform(channel_x_flat)
    channel_y_inv_flat = scalers['y'].inverse_transform(channel_y_flat)
    
    # 4) 重塑回 (N, T)
    channel_x_inv = channel_x_inv_flat.reshape(n_samples, T_samples)
    channel_y_inv = channel_y_inv_flat.reshape(n_samples, T_samples)
    
    # 5) 合并回 (N, T, 2)
    inverse_samples = np.stack([channel_x_inv, channel_y_inv], axis=-1)
    
    return inverse_samples
# ---------------------------------- #H1

def evaluate_samples(samples, scalers, time_stamp=None):
    """
    评估和可视化生成的样本，包括:
    1. 对样本数据进行逆标准化 adaptive_ddpm.py
    2. 可视化几个样本
    3. 计算各通道的协方差矩阵
    4. 计算两个通道间的相关性
    
    参数:
        samples: 生成的样本数据，形状为 (N, T, 2)
        scalers: 用于逆标准化的 scalers 字典 {'x': scaler_x, 'y': scaler_y}
        time_stamp: 用于保存图像的时间戳
    """
    print(">>> Step 6: 评估与可视化")
    
    # 1. 对样本数据进行逆标准化
    print("对样本数据进行逆标准化...")
    samples_inversed = inverse_transform_channels(samples, scalers)
    
    # 2. 可视化几个样本
    print("可视化样本...")
    n_samples_to_plot = 5
    fig, axs = plt.subplots(n_samples_to_plot, 2, figsize=(15, 3*n_samples_to_plot))
    
    for i in range(n_samples_to_plot):
        if i < len(samples_inversed):
            # 绘制第一个通道
            axs[i, 0].plot(samples_inversed[i, :, 0])
            axs[i, 0].set_title(f'样本 {i+1} - 第一通道')
            
            # 绘制第二个通道
            axs[i, 1].plot(samples_inversed[i, :, 1])
            axs[i, 1].set_title(f'样本 {i+1} - 第二通道')
    
    plt.tight_layout()
    if time_stamp:
        plt.savefig(f"diffwave_samples_2d_{time_stamp}.png")
    else:
        plt.savefig(f"diffwave_samples_2d.png")
    plt.close()
    
    # 3. 计算协方差矩阵 - 对每个通道单独计算
    print("计算协方差矩阵...")
    
    # 从样本中提取每个通道
    channel1_samples = samples_inversed[:, :, 0]  # (N, T)
    channel2_samples = samples_inversed[:, :, 1]  # (N, T)
    
    # 计算每个通道的协方差矩阵
    Sigma_ch1 = np.cov(channel1_samples.T)  # (T, T)
    Sigma_ch2 = np.cov(channel2_samples.T)  # (T, T)
    
    # 绘制协方差矩阵
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    
    im1 = axs[0].imshow(Sigma_ch1, cmap='viridis')
    axs[0].set_title('通道1协方差矩阵')
    plt.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(Sigma_ch2, cmap='viridis')
    axs[1].set_title('通道2协方差矩阵')
    plt.colorbar(im2, ax=axs[1])
    
    plt.tight_layout()
    if time_stamp:
        plt.savefig(f"diffwave_covariance_2d_{time_stamp}.png")
    else:
        plt.savefig(f"diffwave_covariance_2d.png")
    plt.close()
    
    # 4. 计算两个通道之间的相关性
    print("计算两通道之间的相关性...")
    
    # 计算每个时间步的两个通道之间的相关系数
    correlations = np.zeros(T)
    for t in range(T):
        correlations[t] = np.corrcoef(channel1_samples[:, t], channel2_samples[:, t])[0, 1]
    
    # 绘制相关系数
    plt.figure(figsize=(10, 6))
    plt.plot(correlations)
    plt.title('两通道间的相关系数（按时间步）')
    plt.xlabel('时间步')
    plt.ylabel('相关系数')
    plt.grid(True)
    if time_stamp:
        plt.savefig(f"diffwave_channel_correlations_{time_stamp}.png")
    else:
        plt.savefig(f"diffwave_channel_correlations.png")
    plt.close()
    
    print("评估完成，已保存可视化结果。")
    return samples_inversed

# ---------------------------------- #H1
def load_and_evaluate(time_stamp):
    """
    从指定时间戳加载模型、样本和标准化器，然后评估和可视化样本。
    
    参数:
        time_stamp: 保存文件的时间戳
    """
    # 加载模型、样本和标准化器
    model, samples, scalers = load_saved_outputs(time_stamp)
    
    # 评估和可视化样本
    print(f"加载时间戳 {time_stamp} 的模型输出并进行评估...")
    inverse_samples = evaluate_samples(samples, scalers, time_stamp)
    
    return model, samples, inverse_samples, scalers
# ---------------------------------- #H1