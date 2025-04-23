import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.special import roots_legendre
from torchinfo import summary


# 设置随机种子以确保可复现性
# torch.manual_seed(42)
# np.random.seed(42)

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


# 2. 定义神经网络模型
class Mish(nn.Module):
    """Mish激活函数: x * tanh(softplus(x))"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))
        # return x * torch.tanh(torch.log(1 + torch.exp(x)))

class ReLU3(nn.Module):
    """ReLU3激活函数: min(max(0,x), 3)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.clamp(torch.relu(x), max=3.0)

# class ResNetBlock(nn.Module):
#     """残差网络块"""
#     def __init__(self, dim, width, activation='mish'):
#         super(ResNetBlock, self).__init__()
#         self.lin1 = nn.Linear(dim, width)
#         self.lin2 = nn.Linear(width, dim)
        
#         # 选择激活函数
#         if activation == 'relu':
#             self.act = nn.ReLU()
#         elif activation == 'mish':
#             self.act = Mish()
#         elif activation == 'relu3':
#             self.act = ReLU3()
#         else:
#             raise ValueError(f"不支持的激活函数: {activation}")

#         # Kaiming 初始化
#         nn.init.kaiming_normal_(self.lin1.weight, mode='fan_out', nonlinearity='relu')
#         nn.init.kaiming_normal_(self.lin2.weight, mode='fan_out', nonlinearity='relu')
#         nn.init.zeros_(self.lin1.bias)
#         nn.init.zeros_(self.lin2.bias)
    
#     def forward(self, x):
#         return x + self.lin2(self.act(self.lin1(x)))

# class ResNet(nn.Module):
#     """残差网络"""
#     def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=6, activation='mish'):
#         super(ResNet, self).__init__()
        
#         self.input_layer = nn.Linear(input_dim, hidden_dim)
        
#         # 选择激活函数
#         if activation == 'relu':
#             self.act = nn.ReLU()
#         elif activation == 'mish':
#             self.act = Mish()
#         elif activation == 'relu3':
#             self.act = ReLU3()
#         else:
#             raise ValueError(f"不支持的激活函数: {activation}")
        
#         self.blocks = nn.ModuleList([
#             ResNetBlock(hidden_dim, hidden_dim, activation) for _ in range(num_blocks)
#         ])
        
#         self.output_layer = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x):
#         x = self.act(self.input_layer(x))
        
#         for block in self.blocks:
#             x = block(x)
        
#         return self.output_layer(x)

class ResNet(nn.Module):
    """根据图片描述修改的 ResNet"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=6, activation='relu'):
        super(ResNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # 输入映射层 - 将输入维度映射到隐藏维度
        self.input_mapping = nn.Linear(input_dim, hidden_dim)
        
        # 每层的线性变换
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_blocks)
        ])
        
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
        
        # Kaiming 初始化
        nn.init.kaiming_normal_(self.input_mapping.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.input_mapping.bias)
        
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        # 初始映射到隐藏维度
        x_mapped = self.act(self.input_mapping(x))
        
        # 初始条件：h_0 = x_mapped, h_{-1} = 0 (都是隐藏维度)
        h = [torch.zeros_like(x_mapped), x_mapped]
        
        # 逐层计算
        for ell in range(self.num_blocks):
            # v_ell = sigma(W_ell * h_{ell-1} + g_ell)
            v_ell = self.act(self.layers[ell](h[-1]))
            # h_ell = h_{ell-2} + v_ell
            h_ell = h[-2] + v_ell  # 现在维度相同，可以直接相加
            h.append(h_ell)
        
        # 最终输出：c^T h_L
        h_L = h[-1]
        return self.output_layer(h_L)

class DriftNet(nn.Module):
    """估计漂移项的网络 - 使用ReLU激活函数"""
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=2, num_blocks=6):
        super(DriftNet, self).__init__()
        self.net = ResNet(input_dim, hidden_dim, output_dim, num_blocks, activation='relu')
    
    def forward(self, x):
        return self.net(x)

# 3. 训练函数

def train_drift_net_from_data(a_nn, x_data, y_data, y_mean, y_std, num_iterations=20000, batch_size=10000, lr=1e-4, device=device):
    """训练漂移项网络 - 使用时间序列数据中的实际位移"""
    print("训练漂移项网络...")
    a_nn.to(device)
    optimizer = optim.Adam(a_nn.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    
    losses = []
    
    
    # 将训练数据预先转移到GPU，减少每次迭代的数据传输开销
    print("将训练数据预加载到GPU...")
    x_data_tensor = torch.tensor(x_data, dtype=torch.float32, device=device)
    y_data_tensor = torch.tensor(y_data, dtype=torch.float32, device=device)
    y_std_tensor = torch.tensor(y_std, dtype=torch.float32, device=device)
    y_mean_tensor = torch.tensor(y_mean, dtype=torch.float32, device=device)
    
    for i in tqdm(range(num_iterations)):
        # 随机抽取批次
        idx = torch.randint(0, len(x_data), (batch_size,), device=device)
        x_batch = x_data_tensor[idx]
        y_batch = y_data_tensor[idx]
        
        # 前向传播 - 模型输出的是归一化空间中的预测
        a_pred = a_nn(x_batch)
        
        # 计算损失：在归一化空间中直接比较
        loss = torch.mean((a_pred - y_batch) ** 2)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(a_nn.parameters(), max_norm=1.0)  # 限制梯度的L2范数不超过1.0
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (i+1) % 1000 == 0:
            print(f"迭代 {i+1}/{num_iterations}, 归一化损失: {loss.item():.6f}")
    
    return losses

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
def evaluate_drift_net(a_nn, grid_points, X, Y, y_mean, y_std):
    """评估漂移项网络
    
    参数:
        a_nn: 训练好的漂移网络模型
        grid_points: 用于可视化的网格点
        X, Y: 用于重构网格的坐标矩阵
        y_mean, y_std: 训练时使用的归一化参数
    """
    # 生成用于计算相对L2误差的10,000个高斯求积点
    eval_points, eval_weights, _, _ = generate_gauss_quadrature_points(nx=100, ny=100, x_range=(-4, 4), y_range=(-6, 6))
    
    # 转换为张量
    x_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    eval_tensor = torch.tensor(eval_points, dtype=torch.float32, device=device)
    
    # 创建用于反归一化的张量
    y_mean_tensor = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_tensor = torch.tensor(y_std, dtype=torch.float32, device=device)
    
    # 计算真实值和预测值（用于可视化）
    with torch.no_grad():
        # 计算真实漂移值
        a_true = drift(x_tensor).cpu().numpy()
        a_pred_normalized = a_nn(x_tensor)
        a_pred = (a_pred_normalized * y_std_tensor + y_mean_tensor).cpu().numpy()
        
        # 为L2误差计算获取评估点的真实值和预测值
        a_true_eval = drift(eval_tensor).cpu().numpy()
        a_pred_normalized_eval = a_nn(eval_tensor)
        a_pred_eval = (a_pred_normalized_eval * y_std_tensor + y_mean_tensor).cpu().numpy()
        
    print(f"应用了反归一化: y_mean={y_mean}, y_std={y_std}")
    
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
    hidden_dim = 50  # 隐藏层维度
    num_blocks = 6   # 残差网络块数量（6层隐藏层）
    data_batch_size = 10000  # 数据批处理大小
    # 训练参数 - 根据论文设置
    num_iterations = 20000  # 训练迭代次数
    drift_batch_size = 10000      # 漂移项批处理大小
    
    # 学习率 - 根据论文设置
    drift_lr = 1e-4        # 漂移项学习率
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
    
    # 转回CPU进行后续处理
    x_data = x_data_tensor.cpu().numpy()
    y_data = y_data_tensor.cpu().numpy()
    
    # 标准化x
    x_mean = x_data.mean(axis=0)
    x_std = x_data.std(axis=0) + 1e-6  # 防止除零
    x_data_normalized = (x_data - x_mean) / x_std

    # 标准化位移数据
    y_mean = y_data.mean(axis=0)
    y_std = y_data.std(axis=0) + 1e-6  # 防止除零
    y_data_normalized = (y_data - y_mean) / y_std
    
    # 显示数据统计信息
    print(f"筛选后数据点数量: {x_data.shape[0]} ({x_data.shape[0]/len(data)*100:.2f}%)")
    # 检查 y_data 的数值范围
    print(f"筛选后数据统计: min={x_data.min(axis=0)}, max={x_data.max(axis=0)}, mean={x_data.mean(axis=0)}, std={x_data.std(axis=0)}")
    print(f"位移数据统计: min={y_data.min(axis=0)}, max={y_data.max(axis=0)}, mean={y_data.mean(axis=0)}, std={y_data.std(axis=0)}")
    print(f"归一化后的位移数据统计: min={y_data_normalized.min(axis=0)}, max={y_data_normalized.max(axis=0)}, mean={y_data_normalized.mean(axis=0)}, std={y_data_normalized.std(axis=0)}")
    
    # 生成用于可视化的网格点
    print("生成网格点用于可视化和评估...")
    vis_grid_points, vis_weights, X, Y = generate_gauss_quadrature_points(nx=100, ny=100, x_range=(-4, 4), y_range=(-6, 6))

    # 初始化漂移网络
    drift_net = DriftNet(input_dim=2, hidden_dim=hidden_dim, output_dim=2, num_blocks=num_blocks).to(device)
    # 打印模型摘要
    summary(drift_net, input_size=(N_filtered, 2))
    
    # 训练和评估模型
    print("\n" + "-"*30)
    print("第一步: 训练漂移项网络")
    print("-"*30)
    drift_losses = train_drift_net_from_data(
        drift_net, 
        x_data_normalized,
        y_data_normalized,
        y_mean,
        y_std,
        num_iterations=num_iterations, 
        batch_size=drift_batch_size, 
        lr=drift_lr
    )
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(drift_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Drift Network Training Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('results/drift_training_loss.png')
    plt.close()
    
    # 评估漂移项网络
    print("\n评估漂移项网络...")
    drift_error = evaluate_drift_net(drift_net, vis_grid_points, X, Y, y_mean, y_std)
    print(f"漂移项网络相对L2误差: {drift_error:.4e}")

# 执行主函数
if __name__ == "__main__":
    main()