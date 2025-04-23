import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.special import roots_legendre

# 设置随机种子以确保可复现性 - 取消注释以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True  # 添加这行使CUDA操作也是确定性的
torch.backends.cudnn.benchmark = False     # 关闭自动优化，确保结果可重现

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def phi(x):
    x1, x2 = x[:, 0], x[:, 1]
    return 1 + (2/15) * (4 * x1**2 - x1 * x2 + x2**2)

# 定义漂移 a(x)
def drift(x):
    x1, x2 = x[:, 0], x[:, 1]
    a1 = -1.5 * x1 + x2
    a2 = 0.25 * x1 - 1.5 * x2
    return torch.stack([a1, a2], dim=1)

# 定义扩散 b(x)
def diffusion(x, d, device):
    batch_size = x.shape[0]
    sqrt_phi_val = torch.sqrt(phi(x))
    
    # 创建一个批处理的2x2矩阵
    b_matrix = torch.zeros(batch_size, d, d, device=device)
    
    # 填充矩阵
    b_matrix[:, 0, 0] = sqrt_phi_val
    b_matrix[:, 0, 1] = 0
    b_matrix[:, 1, 0] = -sqrt_phi_val * (11 / 8)
    b_matrix[:, 1, 1] = (torch.sqrt(torch.tensor(255.0, device=device)) / 8) * sqrt_phi_val
    
    return b_matrix

# 使用批处理和GPU加速的Euler-Maruyama方法
def simulate_sde_batched(N, d, batch_size, delta_t, device):
    # 结果数组，存储在CPU上
    x_result = torch.zeros((N, d), dtype=torch.float32)
    x_result[0] = torch.tensor([0.0, 0.0])  # 初始点
    
    # 当前状态，初始为[0,0]
    x_current = torch.zeros((batch_size, d), device=device)
    
    # 使用批处理进行模拟
    for n in tqdm(range(0, N-1, batch_size)):
        # 确定当前批次的实际大小（最后一批可能小于batch_size）
        current_batch_size = min(batch_size, N-1-n)
        
        if current_batch_size < batch_size:
            x_current = x_current[:current_batch_size]
        
        # 漂移项
        drift_term = drift(x_current) * delta_t
        
        # 扩散项
        diffusion_matrices = diffusion(x_current, d, device)  # 形状: [batch_size, d, d]
        noise = torch.randn(current_batch_size, d, device=device)  # 标准正态噪声
        
        # 对每个样本应用扩散矩阵
        diffusion_term = torch.sqrt(torch.tensor(delta_t, device=device)) * torch.bmm(diffusion_matrices, noise.unsqueeze(-1)).squeeze(-1)
        
        # 更新状态
        x_next = x_current + drift_term + diffusion_term
        
        # 将结果转移到CPU并存储
        x_result[n+1:n+1+current_batch_size] = x_next.cpu()
        
        # 更新当前状态为新状态（用于下一个批次）
        if n + batch_size < N - 1:
            x_current = x_next.clone()
        
    return x_result

# 添加数据标准化类
class Normalizer:
    def __init__(self, data):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)
        self.std[self.std < 1e-8] = 1.0  # 防止除以零
        
    def normalize(self, data):
        return (data - self.mean) / self.std
    
    def denormalize(self, data):
        return data * self.std + self.mean

# 2. 定义神经网络模型
class Mish(nn.Module):
    """Mish激活函数: x * tanh(softplus(x))"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ReLU3(nn.Module):
    """ReLU3激活函数: min(max(0,x), 3)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.clamp(torch.relu(x), max=3.0)

# 改进的ResNet模型
class ResNet(nn.Module):
    """改进的ResNet，优化了结构并添加了批归一化和dropout"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=6, activation='relu', dropout_rate=0.1):
        super(ResNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # 输入映射层：将 input_dim 映射到 hidden_dim
        self.input_map = nn.Linear(input_dim, hidden_dim)
        
        # 残差块
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.ModuleDict({
                'norm1': nn.BatchNorm1d(hidden_dim),
                'linear1': nn.Linear(hidden_dim, hidden_dim),
                'norm2': nn.BatchNorm1d(hidden_dim),
                'linear2': nn.Linear(hidden_dim, hidden_dim),
                'dropout': nn.Dropout(dropout_rate)
            })
            self.blocks.append(block)
        
        # 选择激活函数
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'mish':
            self.act = Mish()
        elif activation == 'relu3':
            self.act = ReLU3()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 更高级的初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        # 为输入映射层初始化
        nn.init.xavier_uniform_(self.input_map.weight)
        nn.init.zeros_(self.input_map.bias)
        
        # 为每个残差块初始化
        for block in self.blocks:
            nn.init.xavier_uniform_(block['linear1'].weight)
            nn.init.zeros_(block['linear1'].bias)
            nn.init.xavier_uniform_(block['linear2'].weight)
            nn.init.zeros_(block['linear2'].bias)
        
        # 为输出层初始化
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        # 将输入映射到 hidden_dim
        h = self.input_map(x)
        
        # 通过残差块
        for block in self.blocks:
            residual = h
            
            # 第一个线性层+激活
            h = block['norm1'](h)
            h = self.act(block['linear1'](h))
            
            # 第二个线性层
            h = block['norm2'](h)
            h = block['linear2'](h)
            
            # 添加残差连接
            h = h + residual
            
            # Dropout
            h = block['dropout'](h)
            
            # 激活
            h = self.act(h)
        
        # 最终输出
        return self.output_layer(h)

class DriftNet(nn.Module):
    """估计漂移项的网络 - 使用改进的ResNet"""
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=2, num_blocks=6, dropout_rate=0.1):
        super(DriftNet, self).__init__()
        self.net = ResNet(input_dim, hidden_dim, output_dim, num_blocks, 
                          activation='relu', dropout_rate=dropout_rate)
    
    def forward(self, x):
        return self.net(x)

# 3. 训练函数
def train_drift_net_from_data(a_nn, x_data, y_data, num_iterations=20000, batch_size=10000, lr=1e-4, 
                             weight_decay=1e-5, device=device, normalizer_x=None, normalizer_y=None):
    """训练漂移项网络 - 使用时间序列数据中的实际位移"""
    print("训练漂移项网络...")
    a_nn.to(device)
    
    # 使用Adam优化器并添加权重衰减 (L2正则化)
    optimizer = optim.Adam(a_nn.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    
    losses = []
    val_losses = []
    
    # 分割数据为训练集和验证集 (90%-10%)
    n_samples = len(x_data)
    n_train = int(0.9 * n_samples)
    
    # 随机置换索引
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # 将训练数据预先转移到GPU
    print("将训练数据预加载到GPU...")
    x_data_tensor = torch.tensor(x_data, dtype=torch.float32, device=device)
    y_data_tensor = torch.tensor(y_data, dtype=torch.float32, device=device)
    
    # 如果提供了归一化器，使用它们
    if normalizer_x is not None and normalizer_y is not None:
        x_data_tensor = normalizer_x.normalize(x_data_tensor)
        y_data_tensor = normalizer_y.normalize(y_data_tensor)
    
    x_train = x_data_tensor[train_indices]
    y_train = y_data_tensor[train_indices]
    x_val = x_data_tensor[val_indices]
    y_val = y_data_tensor[val_indices]
    
    # 提前停止设置
    best_val_loss = float('inf')
    patience = 500  # 提前停止的耐心值
    patience_counter = 0
    best_model_state = None
    
    for i in tqdm(range(num_iterations)):
        # 随机抽取批次进行训练
        idx = torch.randint(0, len(x_train), (batch_size,), device=device)
        x_batch = x_train[idx]
        y_batch = y_train[idx]
        
        # 前向传播
        a_pred = a_nn(x_batch)
        
        # 计算损失
        loss = torch.mean((a_pred - y_batch) ** 2)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        # 启用梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(a_nn.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        # 每100次迭代评估一次验证集
        if (i+1) % 100 == 0:
            with torch.no_grad():
                # 由于验证集可能很大，分批处理
                val_loss = 0
                num_batches = 0
                for j in range(0, len(x_val), batch_size):
                    x_val_batch = x_val[j:j+batch_size]
                    y_val_batch = y_val[j:j+batch_size]
                    
                    val_pred = a_nn(x_val_batch)
                    batch_loss = torch.mean((val_pred - y_val_batch) ** 2).item()
                    val_loss += batch_loss
                    num_batches += 1
                
                val_loss /= num_batches
                val_losses.append(val_loss)
                
                # 检查是否需要保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = a_nn.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 提前停止检查
                if patience_counter >= patience:
                    print(f"提前停止在迭代 {i+1}/{num_iterations}，最佳验证损失: {best_val_loss:.6f}")
                    a_nn.load_state_dict(best_model_state)
                    break
        
        if (i+1) % 1000 == 0:
            print(f"迭代 {i+1}/{num_iterations}, 训练损失: {loss.item():.6f}, 学习率: {scheduler.get_last_lr()[0]:.6e}")
    
    # 如果没有提前停止，加载最佳模型
    if best_model_state is not None and patience_counter < patience:
        a_nn.load_state_dict(best_model_state)
    
    # 绘制训练和验证损失
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Drift Network Training Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(val_losses)*100, 100), val_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.title('Drift Network Validation Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/drift_training_validation_loss.png')
    plt.close()
    
    return losses, val_losses

def generate_gauss_quadrature_points(nx=100, ny=100, x_range=(-4, 4), y_range=(-6, 6)):
    """生成二维高斯求积点
    
    参数:
        nx, ny: 每个维度的高斯点数量
        x_range, y_range: 积分区域的范围
    
    返回:
        points: 形状为(nx*ny, 2)的高斯点坐标
        weights: 对应的积分权重，形状为(nx*ny,)
        X, Y: 重构的网格矩阵，用于可视化
    """
    # 获取一维高斯-勒让德点和权重
    x_points, x_weights = roots_legendre(nx)
    y_points, y_weights = roots_legendre(ny)
    
    # 将[-1,1]范围映射到指定范围
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x_points = 0.5 * (x_max - x_min) * (x_points + 1) + x_min
    y_points = 0.5 * (y_max - y_min) * (y_points + 1) + y_min
    
    # 计算二维点的笛卡尔积
    xx, yy = np.meshgrid(x_points, y_points)
    points = np.column_stack([xx.flatten(), yy.flatten()])
    
    # 计算对应的权重（两个方向权重的张量积）
    xx_weights, yy_weights = np.meshgrid(x_weights, y_weights)
    weights = xx_weights.flatten() * yy_weights.flatten()
    
    # 调整权重，考虑积分区域的大小
    area_factor = 0.25 * (x_max - x_min) * (y_max - y_min)
    weights = weights * area_factor
    
    return points, weights, xx, yy

def compute_relative_L2_error_with_quadrature(func_true, func_pred, grid_points, weights):
    """使用高斯求积计算相对L2误差"""
    true_vals = func_true(grid_points)
    pred_vals = func_pred(grid_points)
    
    # 使用权重计算加权均方误差
    squared_diff = np.sum((true_vals - pred_vals)**2, axis=1)
    error = np.sqrt(np.sum(squared_diff * weights))
    
    # 使用权重计算函数范数
    squared_true = np.sum(true_vals**2, axis=1)
    norm = np.sqrt(np.sum(squared_true * weights))
    
    return error / norm

# 4. 评估函数
def evaluate_drift_net(a_nn, grid_points, X, Y, normalizer_x=None):
    """评估漂移项网络
    
    参数:
        a_nn: 训练好的漂移网络模型
        grid_points: 用于可视化的网格点
        X, Y: 用于重构网格的坐标矩阵
        normalizer_x: 输入数据的归一化器
    """
    # 生成用于计算相对L2误差的10,000个高斯求积点
    eval_points, eval_weights, _, _ = generate_gauss_quadrature_points(nx=100, ny=100, x_range=(-4, 4), y_range=(-6, 6))
    
    # 转换为张量
    x_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    eval_tensor = torch.tensor(eval_points, dtype=torch.float32, device=device)
    
    # 如果提供了归一化器，应用它
    if normalizer_x is not None:
        x_tensor_norm = normalizer_x.normalize(x_tensor)
        eval_tensor_norm = normalizer_x.normalize(eval_tensor)
    else:
        x_tensor_norm = x_tensor
        eval_tensor_norm = eval_tensor

    # 计算真实值和预测值（用于可视化）
    with torch.no_grad():
        # 计算真实漂移值
        a_true = drift(x_tensor).cpu().numpy()
        a_pred = a_nn(x_tensor_norm)
        a_pred = a_pred.cpu().numpy() 
        
        # 为L2误差计算获取评估点的真实值和预测值
        a_true_eval = drift(eval_tensor).cpu().numpy()
        a_pred_eval = a_nn(eval_tensor_norm)
        a_pred_eval = a_pred_eval.cpu().numpy()
    
    # 计算相对L2误差
    error_a1 = np.sqrt(np.mean((a_true_eval[:, 0] - a_pred_eval[:, 0])**2)) / np.sqrt(np.mean(a_true_eval[:, 0]**2))
    error_a2 = np.sqrt(np.mean((a_true_eval[:, 1] - a_pred_eval[:, 1])**2)) / np.sqrt(np.mean(a_true_eval[:, 1]**2))
    total_error = np.sqrt(np.mean(np.sum((a_true_eval - a_pred_eval)**2, axis=1))) / np.sqrt(np.mean(np.sum(a_true_eval**2, axis=1)))
    
    print(f"Drift relative L2 errors - a1: {error_a1:.4e}, a2: {error_a2:.4e}, total: {total_error:.4e}")
    
    # 绘制结果
    plt.figure(figsize=(18, 6))
    
    # 第一个分量
    plt.subplot(131)
    plt.contourf(X, Y, a_true[:, 0].reshape(X.shape), 50, cmap='viridis')
    plt.colorbar(label='True a1')
    plt.title('True Drift a1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.subplot(132)
    plt.contourf(X, Y, a_pred[:, 0].reshape(X.shape), 50, cmap='viridis')
    plt.colorbar(label='Predicted a1')
    plt.title('Predicted Drift a1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.subplot(133)
    plt.contourf(X, Y, (a_true[:, 0] - a_pred[:, 0]).reshape(X.shape), 50, cmap='coolwarm')
    plt.colorbar(label='Error a1')
    plt.title(f'Error in Drift a1 (L2 rel. error: {error_a1:.4e})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.tight_layout()
    plt.savefig('results/drift_a1_comparison.png')
    plt.close()
    
    # 第二个分量
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.contourf(X, Y, a_true[:, 1].reshape(X.shape), 50, cmap='viridis')
    plt.colorbar(label='True a2')
    plt.title('True Drift a2')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.subplot(132)
    plt.contourf(X, Y, a_pred[:, 1].reshape(X.shape), 50, cmap='viridis')
    plt.colorbar(label='Predicted a2')
    plt.title('Predicted Drift a2')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.subplot(133)
    plt.contourf(X, Y, (a_true[:, 1] - a_pred[:, 1]).reshape(X.shape), 50, cmap='coolwarm')
    plt.colorbar(label='Error a2')
    plt.title(f'Error in Drift a2 (L2 rel. error: {error_a2:.4e})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.tight_layout()
    plt.savefig('results/drift_a2_comparison.png')
    plt.close()
    
    return total_error

# 5. 主函数
def main():
    print("\n" + "="*50)
    print("复现论文中的Student's t-distribution实验")
    print("="*50 + "\n")
    
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    
    # 参数设置 - 按照论文要求
    dt = 0.05  # 时间步长
    d = 2   # 维度
    N = 2*10**7  # 数据点数
    hidden_dim = 50  # 隐藏层维度（增加到100）
    num_blocks = 6   # 残差网络块数量（6层隐藏层）
    data_batch_size = 10000  # 数据批处理大小
    
    # 训练参数 - 改进设置
    num_iterations = 20000  # 增加训练迭代次数
    drift_batch_size = 1000  # 降低批次大小，提高泛化性
    
    # 学习率 - 增加初始学习率
    drift_lr = 1e-4  # 漂移项学习率
    
    # L2正则化权重衰减
    weight_decay = 1e-5
    
    # dropout率
    dropout_rate = 0.1
    
    # 定义数据范围
    omega_bounds = ([-4, 4], [-6, 6])  # x1, x2的边界
    
    # 1. 数据生成
    print("生成数据...")
    x = simulate_sde_batched(N, d, data_batch_size, dt, device)
    data = x.numpy()

    # 数据筛选和预处理
    print("筛选和预处理数据...")
    # 首先筛选出在指定范围内的点
    mask_x1 = (data[:, 0] >= omega_bounds[0][0]) & (data[:, 0] <= omega_bounds[0][1]) 
    mask_x2 = (data[:, 1] >= omega_bounds[1][0]) & (data[:, 1] <= omega_bounds[1][1])
    mask = mask_x1 & mask_x2
    filtered_data = data[mask]
    
    # 准备输入和目标数据
    N_filtered = len(filtered_data)
    print(f"筛选后的数据点数量: {N_filtered}")
    
    # 使用CUDA加速计算位移
    print("使用CUDA计算位移...")
    # 将数据转移到GPU
    filtered_data_tensor = torch.tensor(filtered_data, dtype=torch.float32, device=device)
    
    # 使用张量操作计算位移 y = (x_{n+1} - x_n)/dt
    x_data_tensor = filtered_data_tensor[:-1]  # 输入是前N-1个点
    y_data_tensor = (filtered_data_tensor[1:] - filtered_data_tensor[:-1]) / dt
    
    # 创建归一化器
    print("创建归一化器...")
    x_normalizer = Normalizer(x_data_tensor)
    y_normalizer = Normalizer(y_data_tensor)
    
    # 转回CPU进行后续处理
    x_data = x_data_tensor.cpu().numpy()
    y_data = y_data_tensor.cpu().numpy()
    
    # 显示数据统计信息
    print(f"筛选后数据点数量: {x_data.shape[0]} ({x_data.shape[0]/len(data)*100:.2f}%)")
    # 检查 y_data 的数值范围
    print(f"筛选后数据统计: min={x_data.min(axis=0)}, max={x_data.max(axis=0)}, mean={x_data.mean(axis=0)}, std={x_data.std(axis=0)}")
    print(f"位移数据统计: min={y_data.min(axis=0)}, max={y_data.max(axis=0)}, mean={y_data.mean(axis=0)}, std={y_data.std(axis=0)}")
    
    # 生成用于可视化的网格点
    print("生成网格点用于可视化和评估...")
    vis_grid_points, vis_weights, X, Y = generate_gauss_quadrature_points(nx=100, ny=100, x_range=(-4, 4), y_range=(-6, 6))

    # 初始化漂移网络（增加隐藏层大小和添加dropout）
    drift_net = DriftNet(input_dim=2, hidden_dim=hidden_dim, output_dim=2, 
                         num_blocks=num_blocks, dropout_rate=dropout_rate).to(device)
    
    # 训练和评估模型
    print("\n" + "-"*30)
    print("第一步: 训练漂移项网络")
    print("-"*30)
    
    # 使用数据标准化和提前停止训练
    drift_losses, drift_val_losses = train_drift_net_from_data(
        drift_net, 
        x_data,
        y_data,
        num_iterations=num_iterations, 
        batch_size=drift_batch_size, 
        lr=drift_lr,
        weight_decay=weight_decay,
        normalizer_x=x_normalizer,
        normalizer_y=y_normalizer
    )
    
    # 评估漂移项网
    # 评估漂移项网络
    print("\n评估漂移项网络...")
    drift_error = evaluate_drift_net(drift_net, vis_grid_points, X, Y, normalizer_x=x_normalizer)
    print(f"漂移项网络相对L2误差: {drift_error:.4e}")
    
    # 保存训练好的模型
    print("保存模型...")
    torch.save({
        'model_state_dict': drift_net.state_dict(),
        'normalizer_x_mean': x_normalizer.mean.cpu(),
        'normalizer_x_std': x_normalizer.std.cpu(),
        'normalizer_y_mean': y_normalizer.mean.cpu(),
        'normalizer_y_std': y_normalizer.std.cpu(),
    }, 'results/drift_net_model.pth')
    
    # 可视化训练过程中的预测情况
    print("可视化训练效果...")
    plt.figure(figsize=(15, 10))
    
    # 选择几个随机点进行可视化
    vis_indices = np.random.choice(len(x_data), min(1000, len(x_data)), replace=False)
    x_vis = torch.tensor(x_data[vis_indices], dtype=torch.float32, device=device)
    y_vis = torch.tensor(y_data[vis_indices], dtype=torch.float32, device=device)
    
    # 使用网络进行预测
    x_vis_norm = x_normalizer.normalize(x_vis)
    with torch.no_grad():
        y_pred = drift_net(x_vis_norm).cpu().numpy()
    
    y_vis = y_vis.cpu().numpy()
    
    # 可视化第一个分量
    plt.subplot(221)
    plt.scatter(y_vis[:, 0], y_pred[:, 0], alpha=0.3)
    min_val = min(y_vis[:, 0].min(), y_pred[:, 0].min())
    max_val = max(y_vis[:, 0].max(), y_pred[:, 0].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True a1')
    plt.ylabel('Predicted a1')
    plt.title('Drift Component a1')
    plt.axis('square')
    plt.grid(True)
    
    # 可视化第二个分量
    plt.subplot(222)
    plt.scatter(y_vis[:, 1], y_pred[:, 1], alpha=0.3)
    min_val = min(y_vis[:, 1].min(), y_pred[:, 1].min())
    max_val = max(y_vis[:, 1].max(), y_pred[:, 1].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True a2')
    plt.ylabel('Predicted a2')
    plt.title('Drift Component a2')
    plt.axis('square')
    plt.grid(True)
    
    # 计算并显示每个分量的相关性
    corr_a1 = np.corrcoef(y_vis[:, 0], y_pred[:, 0])[0, 1]
    corr_a2 = np.corrcoef(y_vis[:, 1], y_pred[:, 1])[0, 1]
    
    plt.subplot(223)
    plt.text(0.5, 0.5, f"Correlation for a1: {corr_a1:.4f}\nCorrelation for a2: {corr_a2:.4f}", 
             ha='center', va='center', fontsize=14)
    plt.axis('off')
    
    # 计算并显示整体误差
    mse_a1 = np.mean((y_vis[:, 0] - y_pred[:, 0])**2)
    mse_a2 = np.mean((y_vis[:, 1] - y_pred[:, 1])**2)
    relative_error_a1 = np.sqrt(mse_a1) / np.std(y_vis[:, 0])
    relative_error_a2 = np.sqrt(mse_a2) / np.std(y_vis[:, 1])
    
    plt.subplot(224)
    plt.text(0.5, 0.5, f"MSE for a1: {mse_a1:.4e}\nMSE for a2: {mse_a2:.4e}\n" +
             f"Relative error a1: {relative_error_a1:.4e}\nRelative error a2: {relative_error_a2:.4e}", 
             ha='center', va='center', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/drift_prediction_scatter.png')
    plt.close()
    
    # 绘制样本轨迹与预测比较
    print("绘制样本轨迹比较...")
    
    # 从原始数据中抽取一段轨迹
    traj_start = np.random.randint(0, len(filtered_data) - 1000)
    traj_length = 1000
    traj_data = filtered_data[traj_start:traj_start+traj_length]
    
    # 计算原始和预测的变化率
    traj_x = torch.tensor(traj_data[:-1], dtype=torch.float32, device=device)
    traj_y_true = (traj_data[1:] - traj_data[:-1]) / dt
    
    traj_x_norm = x_normalizer.normalize(traj_x)
    with torch.no_grad():
        traj_y_pred = drift_net(traj_x_norm).cpu().numpy()
    
    # 绘制轨迹比较
    plt.figure(figsize=(15, 10))
    time_steps = np.arange(len(traj_data)-1) * dt
    
    plt.subplot(221)
    plt.plot(time_steps, traj_data[:-1, 0], label='Position x1')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Trajectory Position x1')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(222)
    plt.plot(time_steps, traj_data[:-1, 1], label='Position x2')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Trajectory Position x2')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(223)
    plt.plot(time_steps, traj_y_true[:, 0], label='True velocity')
    plt.plot(time_steps, traj_y_pred[:, 0], label='Predicted velocity', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Trajectory Velocity x1')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(224)
    plt.plot(time_steps, traj_y_true[:, 1], label='True velocity')
    plt.plot(time_steps, traj_y_pred[:, 1], label='Predicted velocity', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Trajectory Velocity x2')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/trajectory_comparison.png')
    plt.close()
    
    print("\n实验完成！结果已保存到 results/ 目录")

# 执行主函数
if __name__ == "__main__":
    main()