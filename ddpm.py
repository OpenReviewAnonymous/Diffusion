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
import logging

logger = None

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 时间步数和样本数量
T_diffusion = 100
N = 100 * 1000

# Assuming time series length is the same as diffusion steps based on original code
T_timeseries = T_diffusion

# 配置参数
# DDPM_LR = 1e-4 # Original fixed value
# 从环境变量或使用默认值读取学习率
DDPM_LR = float(os.environ.get("HP_DDPM_LR", "1e-4")) # Read from environment variable HP_DDPM_LR
DDPM_LR_ETAMIN = 1e-5
# BATCH_SIZE = 64
BATCH_SIZE = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================
# Adaptive DDPM Model Definition
# ======================
class AdaptiveDDPM(nn.Module):
    # ---------------------------------- #A3
    # Add D=2 to handle dual channels, input_dim = T * D = 2T
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
        # TODO This value is also a hyperparameter that can be adjusted

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

# ----------------- DDPM Model Definition -----------------
class DDPM(nn.Module):
    def __init__(self, T):
        super(DDPM, self).__init__()
        self.embedding_dim = 32

        # Time embedding
        self.time_embed = nn.Embedding(T, self.embedding_dim)

        # Main network structure
        self.model = nn.Sequential(
            nn.Linear(T + self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, T)
        )

    def get_noise(self, batch_size):
        """Generate noise"""
        return torch.randn(batch_size, self.T_diffusion, device=self.time_embed.weight.device)

    def forward(self, x, t):
        """Forward pass"""
        batch_size = x.size(0)
        t_embed = self.time_embed(t)
        x_with_t = torch.cat([x, t_embed], dim=1)
        out = self.model(x_with_t)
        return out

# 工具函数
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def setup_parameters(config):
    """步骤1：设置参数"""
    global logger
    logger.info(">>> Step 1: 设置参数")
    # 设置噪声调度
    betas = torch.linspace(1e-4, 0.02, T_diffusion, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # 添加到配置
    config.update({
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars
    })
    
    return config

def load_data():
    """步骤2：加载数据"""
    global logger
    logger.info(">>> Step 2: 加载数据")
    # 尝试从文件加载2D数据
    try:
        DATASET_PATH = os.environ.get("DATASET_PATH", "two_dimension_garch.npy")
        volumes = np.load(DATASET_PATH)
        logger.info(f"已加载2D数据，形状: {volumes.shape}")
        logger.info(f"DATASET_PATH: {DATASET_PATH}")
        
        N_all, T_data, D_data = volumes.shape
        volumes_filtered = volumes.copy()

        logger.info(f"原始形状: {volumes.shape}")
        volumes_filtered = volumes_filtered[:int(N*1.2), :T_timeseries, :]
        logger.info(f"处理后形状: {volumes_filtered.shape}")
        
        S_true = volumes_filtered
        D = 2  # 2D数据

    except FileNotFoundError:
        raise FileNotFoundError("2D数据文件未找到，生成AR(1)-GARCH(1,1)数据...")
        # logger.info("2D数据文件未找到，生成AR(1)-GARCH(1,1)数据...")
        # # 生成1D数据
        # mu = 0
        # phi = 0.95  
        # omega = 0.05
        # alpha_garch = 0.05
        # beta_garch = 0.90

        # S_true = np.zeros((N, T_timeseries))
        # for n in range(N):
        #     x = np.zeros(T_timeseries)
        #     epsilon = np.zeros(T_timeseries)
        #     sigma2 = np.zeros(T_timeseries)
        #     z = np.random.randn(T_timeseries)
        #     sigma2[0] = omega / (1 - alpha_garch - beta_garch)  # 初始化方差
        #     epsilon[0] = np.sqrt(sigma2[0]) * z[0]
        #     x[0] = mu + epsilon[0]
        #     for t in range(1, T_timeseries):
        #         sigma2[t] = omega + alpha_garch * epsilon[t-1]**2 + beta_garch * sigma2[t-1]
        #         epsilon[t] = np.sqrt(sigma2[t]) * z[t]
        #         x[t] = mu + phi * x[t-1] + epsilon[t]
        #     S_true[n, :] = x
        # D = 1  # 1D数据

    # 划分数据集
    total_samples = len(S_true)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_data = S_true[:train_size]
    val_data = S_true[train_size:train_size+val_size]
    test_data = S_true[train_size+val_size:]

    # 标准化处理
    if D == 2:
        # 2D数据处理
        train_X = train_data[..., 0]  # shape (train_size, T_timeseries)
        train_Y = train_data[..., 1]  # shape (train_size, T_timeseries)
        val_X = val_data[..., 0]  
        val_Y = val_data[..., 1]
        test_X = test_data[..., 0]  
        test_Y = test_data[..., 1]

        # 展平为 (N*T_timeseries, 1)，对每个通道单独标准化
        train_X_flat = train_X.reshape(-1, 1)
        train_Y_flat = train_Y.reshape(-1, 1)
        val_X_flat = val_X.reshape(-1, 1)
        val_Y_flat = val_Y.reshape(-1, 1)
        test_X_flat = test_X.reshape(-1, 1)
        test_Y_flat = test_Y.reshape(-1, 1)

        # 为X、Y分别创建scaler
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        # 只在训练集上fit
        train_X_scaled_flat = scaler_x.fit_transform(train_X_flat)
        train_Y_scaled_flat = scaler_y.fit_transform(train_Y_flat)

        # 验证集与测试集只transform
        val_X_scaled_flat = scaler_x.transform(val_X_flat)
        val_Y_scaled_flat = scaler_y.transform(val_Y_flat)
        test_X_scaled_flat = scaler_x.transform(test_X_flat)
        test_Y_scaled_flat = scaler_y.transform(test_Y_flat)

        # 重塑回 (N, T_timeseries)
        train_X_scaled = train_X_scaled_flat.reshape(train_size, T_timeseries)
        train_Y_scaled = train_Y_scaled_flat.reshape(train_size, T_timeseries)
        val_X_scaled = val_X_scaled_flat.reshape(val_size, T_timeseries)
        val_Y_scaled = val_Y_scaled_flat.reshape(val_size, T_timeseries)
        test_X_scaled = test_X_scaled_flat.reshape(test_size, T_timeseries)
        test_Y_scaled = test_Y_scaled_flat.reshape(test_size, T_timeseries)

        # 合并回 (N, T_timeseries, 2)
        train_scaled = np.stack([train_X_scaled, train_Y_scaled], axis=-1)
        val_scaled = np.stack([val_X_scaled, val_Y_scaled], axis=-1)
        test_scaled = np.stack([test_X_scaled, test_Y_scaled], axis=-1)

        # 打包标准化器
        scalers = {
            'x': scaler_x,
            'y': scaler_y
        }
    else:
        # 1D数据处理
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        val_scaled = scaler.transform(val_data)
        test_scaled = scaler.transform(test_data)
        
        scalers = scaler

    # 转换为tensor
    train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
    val_tensor = torch.tensor(val_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

    # 创建DataLoader
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    test_dataset = TensorDataset(test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loader = (train_loader, val_loader, test_loader)
    return data_loader, scalers, test_size, D

def setup_model(num_epochs, D):
    """步骤3：设置模型"""
    global logger
    logger.info(">>> Step 3: 设置模型")
    model_ddpm = DDPM(T_diffusion=T_diffusion).to(device)
    
    optimizer_ddpm = AdamW(model_ddpm.parameters(), lr=DDPM_LR)
    loss_fn = nn.MSELoss()
    logger.info(f"loss_fn: {loss_fn}")
    
    scheduler_ddpm = CosineAnnealingLR(optimizer_ddpm, T_max=num_epochs, eta_min=DDPM_LR_ETAMIN)
    
    return model_ddpm, optimizer_ddpm, scheduler_ddpm, loss_fn

def train_ddpm(model, optimizer, scheduler, data_loader, num_epochs, time_stamp, alpha_bars, loss_fn):
    """步骤4：训练模型"""
    global logger
    logger.info(">>> Step 4: 开始训练")
    train_loader, val_loader, _ = data_loader
    best_val_loss = float('inf')
    model.train()
    
    # 初始化TensorBoard
    writer = SummaryWriter(f'runs/ddpm_{time_stamp}')
    
    for epoch in tqdm(range(num_epochs)):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_loader:
            x0 = batch[0].to(device)
            batch_size_ = x0.size(0)
            t = torch.randint(0, T_diffusion, (batch_size_,), device=device)
            
            epsilon = torch.randn_like(x0)
            xt = torch.sqrt(alpha_bars[t].view(-1, 1)) * x0 + torch.sqrt(1 - alpha_bars[t].view(-1, 1)) * epsilon
            epsilon_theta = model(xt, t)
            loss = loss_fn(epsilon_theta, epsilon)
            
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x0 = batch[0].to(device)
                batch_size_ = x0.size(0)
                t = torch.randint(0, T_diffusion, (batch_size_,), device=device)
                
                epsilon = torch.randn_like(x0)
                xt = torch.sqrt(alpha_bars[t].view(-1, 1)) * x0 + torch.sqrt(1 - alpha_bars[t].view(-1, 1)) * epsilon
                epsilon_theta = model(xt, t)
                loss = loss_fn(epsilon_theta, epsilon)
                val_loss += loss.item()
        
        scheduler.step()
        
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        # 记录指标到TensorBoard
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Loss/train', train_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/train: {train_loss:.6f}")
        writer.add_scalar('Loss/validation', val_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Loss/validation: {val_loss:.6f}")
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - TB Learning_Rate: {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存最佳模型
            torch.save(model.state_dict(), f"best_model_ddpm_{time_stamp}.pth")
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'[DDPM] Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')
    
    writer.close()
    return model

def inference(model, N_test, scalers, config):
    """步骤5：采样推理"""
    global logger
    logger.info(">>> Step 5: 采样推理")
    model.eval()
    betas = config['betas']
    alphas = config['alphas']
    alpha_bars = config['alpha_bars']
    
    with torch.no_grad():
        # 初始化噪声
        x_t = model.get_noise(N_test).to(device)
        
        # 逆向扩散过程
        for t_ in tqdm(reversed(range(model.T_diffusion)), desc="Sampling"):
            t_batch = torch.full((N_test,), t_, device=device)
            epsilon_theta = model(x_t, t_batch)
            
            alpha_t = alphas[t_]
            alpha_bar_t = alpha_bars[t_]
            
            if t_ > 0:
                sigma_t = torch.sqrt(betas[t_])
                z = model.get_noise(N_test).to(device)
            else:
                sigma_t = 0
                z = 0
            
            # 逆向扩散步骤
            x_t = (x_t - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t)*epsilon_theta) / torch.sqrt(alpha_t)
            x_t += sigma_t * z
        
        samples = x_t.cpu().numpy()
    
    return samples

def inverse_transform_channels(samples, scalers, D):
    """将生成的样本从标准化空间转换回原始空间"""
    if D == 1:
        if isinstance(scalers, dict):
            # 兼容2D处理方式
            return scalers['x'].inverse_transform(samples)
        else:
            return scalers.inverse_transform(samples)
    else:
        # 2D数据
        n_samples = samples.shape[0]
        T_samples = samples.shape[1]
        
        # 分离两个通道
        channel_x = samples[..., 0]  # (N, T_timeseries)
        channel_y = samples[..., 1]  # (N, T_timeseries)
        
        # 重塑为 (N*T_timeseries, 1) 以便应用 inverse_transform
        channel_x_flat = channel_x.reshape(-1, 1)
        channel_y_flat = channel_y.reshape(-1, 1)
        
        # 分别应用逆标准化
        channel_x_inv_flat = scalers['x'].inverse_transform(channel_x_flat)
        channel_y_inv_flat = scalers['y'].inverse_transform(channel_y_flat)
        
        # 重塑回 (N, T_timeseries)
        channel_x_inv = channel_x_inv_flat.reshape(n_samples, T_timeseries)
        channel_y_inv = channel_y_inv_flat.reshape(n_samples, T_timeseries)
        
        # 合并回 (N, T_timeseries, 2)
        inverse_samples = np.stack([channel_x_inv, channel_y_inv], axis=-1)
        
        return inverse_samples

def visualize_samples(samples, D, time_stamp):
    """可视化生成的样本"""
    global logger
    logger.info(">>> Step 6: 可视化样本")
    
    # 创建保存目录
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    n_samples_to_plot = 5
    
    if D == 1:
        plt.figure(figsize=(15, 10))
        for i in range(n_samples_to_plot):
            plt.subplot(n_samples_to_plot, 1, i+1)
            plt.plot(samples[i])
            plt.title(f'样本 {i+1}')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ddpm_samples_1d_{time_stamp}.png")
        plt.close()
        
        # 计算协方差矩阵
        cov_matrix = np.cov(samples[:100].reshape(100, -1).T)
        plt.figure(figsize=(10, 8))
        plt.imshow(cov_matrix, cmap="viridis")
        plt.colorbar(label="协方差")
        plt.title("生成样本的协方差矩阵")
        plt.savefig(f"{save_dir}/ddpm_covariance_1d_{time_stamp}.png")
        plt.close()
    
    else:  # 2D数据
        fig, axs = plt.subplots(n_samples_to_plot, 2, figsize=(15, 3*n_samples_to_plot))
        
        for i in range(n_samples_to_plot):
            if i < len(samples):
                # 绘制第一个通道
                axs[i, 0].plot(samples[i, :, 0])
                axs[i, 0].set_title(f'样本 {i+1} - 第一通道')
                
                # 绘制第二个通道
                axs[i, 1].plot(samples[i, :, 1])
                axs[i, 1].set_title(f'样本 {i+1} - 第二通道')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ddpm_samples_2d_{time_stamp}.png")
        plt.close()
        
        # 计算各通道的协方差矩阵
        channel1_samples = samples[:100, :, 0]  # (N, T_timeseries)
        channel2_samples = samples[:100, :, 1]  # (N, T_timeseries)
        
        Sigma_ch1 = np.cov(channel1_samples.T)  # (T_timeseries, T_timeseries)
        Sigma_ch2 = np.cov(channel2_samples.T)  # (T_timeseries, T_timeseries)
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        
        im1 = axs[0].imshow(Sigma_ch1, cmap='viridis')
        axs[0].set_title('通道1协方差矩阵')
        plt.colorbar(im1, ax=axs[0])
        
        im2 = axs[1].imshow(Sigma_ch2, cmap='viridis')
        axs[1].set_title('通道2协方差矩阵')
        plt.colorbar(im2, ax=axs[1])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ddpm_covariance_2d_{time_stamp}.png")
        plt.close()
        
        # 计算两个通道之间的相关性
        correlations = np.zeros(T_timeseries)
        for t in range(T_timeseries):
            correlations[t] = np.corrcoef(channel1_samples[:, t], channel2_samples[:, t])[0, 1]
        
        plt.figure(figsize=(10, 6))
        plt.plot(correlations)
        plt.title('两通道间的相关系数（按时间步）')
        plt.xlabel('时间步')
        plt.ylabel('相关系数')
        plt.grid(True)
        plt.savefig(f"{save_dir}/ddpm_channel_correlations_{time_stamp}.png")
        plt.close()

def get_timestamp():
    """生成时间戳"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

def run_ddpm():
    """主函数：运行DDPM模型的完整流程"""
    # 步骤0：设置时间戳
    config = {
        'num_epochs': 500,
        'T_diffusion': T_diffusion,
        'T_timeseries': T_timeseries,
        'device': device,
        'time_stamp': get_timestamp()
    }
    
    # Configure logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"ddpm_{config['time_stamp']}.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler() # 同时输出到控制台
                        ])
    global logger
    logger = logging.getLogger(__name__)

    # Log initial run parameters and hyperparameters after logger is initialized
    logger.info(f">>> Step 0: 设置时间戳 {config['time_stamp']}")
    logger.info(f"T_diffusion: {T_diffusion}")
    logger.info(f"T_timeseries: {T_timeseries}")
    logger.info(f"N (Total Samples): {N}")
    logger.info(f"DDPM_LR: {DDPM_LR}")
    logger.info(f"DDPM_LR_ETAMIN: {DDPM_LR_ETAMIN}")
    logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"Using device: {device}")
    logger.info(f"Num Epochs: {config['num_epochs']}")
    logger.info(f"Run ID: {config.get('run_id', 'N/A')}")
    logger.info(f"Time Stamp: {config['time_stamp']}")

    # 步骤1：设置参数
    config = setup_parameters(config)
    
    # 步骤2：加载数据
    data_loader, scalers, test_size, D = load_data()
    
    # 步骤3：设置模型
    model_ddpm, optimizer_ddpm, scheduler_ddpm, loss_fn = setup_model(config['num_epochs'], D)
    
    # 步骤4：训练模型
    model_ddpm = train_ddpm(
        model_ddpm, optimizer_ddpm, scheduler_ddpm, 
        data_loader, config['num_epochs'], config['time_stamp'], 
        config['alpha_bars'], loss_fn
    )
    
    # 步骤5：采样推理
    samples = inference(
        model_ddpm, test_size, scalers, config
    )
    
    # 反标准化
    samples_inversed = inverse_transform_channels(samples, scalers, D)
    
    # 步骤6：可视化
    visualize_samples(samples_inversed, D, config['time_stamp'])
    
    # 保存结果
    save_dir = f"results_ddpm_{config['time_stamp']}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/samples.npy", samples)
    np.save(f"{save_dir}/samples_inversed.npy", samples_inversed)
    
    # 保存标准化器
    if D == 2:
        joblib.dump(scalers, f"{save_dir}/scalers.joblib")
    else:
        joblib.dump(scalers, f"{save_dir}/scaler.joblib")
    
    logger.info(f"结果已保存至 {save_dir}")
    return data_loader, model_ddpm, samples_inversed, scalers, config['time_stamp']

# 仅当直接运行此脚本时执行
if __name__ == "__main__":
    run_ddpm()