import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置随机种子以确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 数据生成函数 - 使用EM方案生成时间序列
def phi(x1, x2):
    """计算phi(x1, x2) = 1 + (2/15)(4x1^2 - x1*x2 + x2^2)"""
    return 1 + (2/15) * (4*x1**2 - x1*x2 + x2**2)

def drift(x):
    """漂移项 a(x)"""
    x1, x2 = x[:, 0], x[:, 1]
    a1 = -1.5*x1 + x2
    a2 = 0.25*x1 - 1.5*x2
    return torch.stack([a1, a2], dim=1)

def diffusion(x):
    """扩散矩阵 b(x)"""
    x1, x2 = x[:, 0], x[:, 1]
    phi_val = phi(x1, x2)
    sqrt_phi = torch.sqrt(phi_val)
    
    b11 = sqrt_phi
    b12 = torch.zeros_like(x1)
    b21 = -11/8 * sqrt_phi
    b22 = np.sqrt(255)/8 * sqrt_phi
    
    # 返回形状为 [batch_size, 2, 2] 的张量
    return torch.stack([
        torch.stack([b11, b12], dim=1),
        torch.stack([b21, b22], dim=1)
    ], dim=1)

def diffusion_product(x):
    """计算扩散矩阵的乘积 b(x)b(x)^T"""
    b = diffusion(x)
    batch_size = x.shape[0]
    bbt = torch.zeros(batch_size, 2, 2, device=x.device)
    
    for i in range(batch_size):
        bbt[i] = torch.matmul(b[i], b[i].t())
    
    return bbt

def true_density(x1, x2):
    """真实的平稳密度"""
    p = 2 / (np.pi * np.sqrt(15)) * (phi(x1, x2) ** (-3))
    return p

def generate_data(dt=0.05, N=2*10**7, x0=None):
    """使用Euler-Maruyama方案生成时间序列数据"""
    print(f"Generating {N} data points with dt={dt}...")
    
    if x0 is None:
        x0 = np.zeros(2)  # 初始点
    
    x = np.zeros((N, 2))
    x[0] = x0
    
    for n in tqdm(range(N-1)):
        # 当前位置
        xi = x[n]
        xi_tensor = torch.tensor([xi], dtype=torch.float32)
        
        # 计算漂移项
        a_val = drift(xi_tensor).numpy()[0]
        
        # 计算扩散矩阵
        b_val = diffusion(xi_tensor).numpy()[0]
        
        # 生成随机增量
        dW = np.random.normal(0, np.sqrt(dt), 2)
        
        # 更新位置
        x[n+1] = xi + a_val * dt + np.dot(b_val, dW)
        
        # 每隔一定步数保存进度
        if (n+1) % (N//10) == 0:
            print(f"Generated {n+1}/{N} data points ({(n+1)/N*100:.1f}%)")
    
    # 过滤在范围 [-4, 4]×[-6, 6] 内的点
    mask = (x[:, 0] >= -4) & (x[:, 0] <= 4) & (x[:, 1] >= -6) & (x[:, 1] <= 6)
    x_filtered = x[mask]
    
    print(f"Filtered data: {x_filtered.shape[0]} points ({x_filtered.shape[0]/N*100:.2f}%)")
    return x_filtered

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

class ResNetBlock(nn.Module):
    """残差网络块"""
    def __init__(self, dim, width, activation='mish'):
        super(ResNetBlock, self).__init__()
        self.lin1 = nn.Linear(dim, width)
        self.lin2 = nn.Linear(width, dim)
        
        # 选择激活函数
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'mish':
            self.act = Mish()
        elif activation == 'relu3':
            self.act = ReLU3()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # Kaiming 初始化
        nn.init.kaiming_normal_(self.lin1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.lin2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.lin1.bias)
        nn.init.zeros_(self.lin2.bias)
    
    def forward(self, x):
        return x + self.lin2(self.act(self.lin1(x)))

class ResNet(nn.Module):
    """残差网络"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=6, activation='mish'):
        super(ResNet, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # 选择激活函数
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'mish':
            self.act = Mish()
        elif activation == 'relu3':
            self.act = ReLU3()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, hidden_dim, activation) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.act(self.input_layer(x))
        
        for block in self.blocks:
            x = block(x)
        
        return self.output_layer(x)

class DriftNet(nn.Module):
    """估计漂移项的网络 - 使用ReLU激活函数"""
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=2, num_blocks=6):
        super(DriftNet, self).__init__()
        self.net = ResNet(input_dim, hidden_dim, output_dim, num_blocks, activation='relu')
    
    def forward(self, x):
        return self.net(x)

class DiffusionNet(nn.Module):
    """估计扩散矩阵的网络 - 使用Mish激活函数"""
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=3, num_blocks=6):
        super(DiffusionNet, self).__init__()
        # 输出三个值：BB^T的三个独特元素 (因为是对称矩阵)
        self.net = ResNet(input_dim, hidden_dim, output_dim, num_blocks, activation='mish')
    
    def forward(self, x):
        # 输出BB^T的三个独特元素
        out = self.net(x)
        
        batch_size = x.shape[0]
        BBT = torch.zeros(batch_size, 2, 2, device=x.device)
        
        # 构建2x2对称矩阵
        BBT[:, 0, 0] = torch.exp(out[:, 0])  # 确保对角元素为正
        BBT[:, 1, 1] = torch.exp(out[:, 1])  # 确保对角元素为正
        BBT[:, 0, 1] = out[:, 2]
        BBT[:, 1, 0] = out[:, 2]
        
        return BBT

class DensityNet(nn.Module):
    """估计平稳密度的网络 - 使用ReLU3激活函数"""
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_blocks=6):
        super(DensityNet, self).__init__()
        self.net = ResNet(input_dim, hidden_dim, output_dim, num_blocks, activation='relu3')
    
    def forward(self, x):
        # 确保密度为正
        return torch.exp(self.net(x))

# 3. 训练函数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_drift_net_from_data(a_nn, train_data, dt=0.05, num_iterations=20000, batch_size=10000, lr=1e-4, omega_bounds=([-4, 4], [-6, 6])):
    """训练漂移项网络 - 使用时间序列数据中的实际位移"""
    print("训练漂移项网络...")
    a_nn.to(device)
    optimizer = optim.Adam(a_nn.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    
    losses = []
    
    # 首先计算所有相邻时间步的位移
    N = len(train_data)
    y_data = np.zeros((N-1, train_data.shape[1]))
    for n in range(N-1):
        y_data[n] = (train_data[n+1] - train_data[n]) / dt
    
    # 标准化位移数据
    y_mean = y_data.mean(axis=0)
    y_std = y_data.std(axis=0) + 1e-6  # 防止除零
    y_data_normalized = (y_data - y_mean) / y_std
    
    # 然后仅保留输入点在[-4,4]×[-6,6]范围内的数据对
    x_data = train_data[:-1]  # 输入点
    mask_x1 = (x_data[:, 0] >= -4) & (x_data[:, 0] <= 4) 
    mask_x2 = (x_data[:, 1] >= -6) & (x_data[:, 1] <= 6)
    mask = mask_x1 & mask_x2
    
    x_data = x_data[mask]
    y_data = y_data[mask]
    
    # 显示数据统计信息
    print("\n筛选后的数据统计信息:")
    print(f"数据点数量: {x_data.shape[0]} ({x_data.shape[0]/len(train_data)*100:.2f}%)")
    print(f"最小值: {x_data.min(axis=0)}")
    print(f"最大值: {x_data.max(axis=0)}")
    print(f"均值: {x_data.mean(axis=0)}")
    print(f"标准差: {x_data.std(axis=0)}")

    print(f"训练数据准备完成: x_data: {x_data.shape}, y_data: {y_data.shape}")
    
    # 检查 y_data 的数值范围
    print(f"位移数据统计: min={y_data.min(axis=0)}, max={y_data.max(axis=0)}, mean={y_data.mean(axis=0)}, std={y_data.std(axis=0)}")
    
    for i in tqdm(range(num_iterations)):
        # 随机抽取批次
        idx = np.random.choice(len(x_data), batch_size)
        x_batch = torch.tensor(x_data[idx], dtype=torch.float32, device=device)
        y_batch = torch.tensor(y_data_normalized[idx], dtype=torch.float32, device=device)
        
        # 前向传播
        a_pred = a_nn(x_batch)
        
        # 反归一化预测值
        y_std_tensor = torch.tensor(y_std, dtype=torch.float32, device=device)
        y_mean_tensor = torch.tensor(y_mean, dtype=torch.float32, device=device)
        a_pred_denormalized = a_pred * y_std_tensor + y_mean_tensor
        
        # 计算损失：直接计算所有维度的 MSE
        loss = torch.mean((a_pred_denormalized - y_batch) ** 2)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (i+1) % 1000 == 0:
            print(f"迭代 {i+1}/{num_iterations}, 损失: {loss.item():.6f}")
    
    return losses

def train_diffusion_net(b_nn, train_data, num_iterations=20000, batch_size=10000, lr=1e-4):
    """训练扩散矩阵网络"""
    print("Training diffusion network...")
    optimizer = optim.Adam(b_nn.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    
    losses = []
    
    for i in tqdm(range(num_iterations)):
        # 随机抽取批次
        idx = np.random.choice(len(train_data), batch_size)
        x_batch = torch.tensor(train_data[idx], dtype=torch.float32, device=device)
        
        # 计算真实扩散矩阵乘积
        bbt_true = diffusion_product(x_batch)
        
        # 前向传播
        bbt_pred = b_nn(x_batch)
        
        # 计算损失
        loss = torch.mean((bbt_pred - bbt_true)**2)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if (i+1) % 1000 == 0:
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")
    
    return losses

def generate_grid_points(nx=300, ny=300, for_eval=False):
    """生成评估网格点
    
    参数:
        nx, ny: 网格点数量
        for_eval: 如果为True，则生成10,000个点（用于计算相对L2误差）
    """
    if for_eval:
        # 为了计算相对L2误差，生成10,000个高斯求积点
        # 这里简单地生成均匀网格点，实际上应该使用高斯求积点
        num_points = 10000
        x_points = np.random.uniform(-4, 4, num_points)
        y_points = np.random.uniform(-6, 6, num_points)
        points = np.column_stack([x_points, y_points])
        return points, None, None
    else:
        # 为可视化生成网格点
        x = np.linspace(-4, 4, nx)
        y = np.linspace(-6, 6, ny)
        X, Y = np.meshgrid(x, y)
        
        # 将网格点展平为坐标点列表
        points = np.column_stack([X.ravel(), Y.ravel()])
        return points, X, Y

def compute_relative_L2_error(func_true, func_pred, grid_points):
    """计算相对L2误差，使用10,000个高斯求积点"""
    true_vals = func_true(grid_points)
    pred_vals = func_pred(grid_points)
    
    error = np.sqrt(np.mean((true_vals - pred_vals)**2))
    norm = np.sqrt(np.mean(true_vals**2))
    
    return error / norm

def train_density_net(p_nn, a_nn, b_nn, train_data, grid_points, 
                      num_iterations=20000, batch_size_data=10000, 
                      batch_size_boundary=4000, batch_size_quadrature=300**2,
                      lr=1e-3, lambda_param=1, gamma_param=500):
    """训练密度估计网络"""
    print("Training density network...")
    p_nn.to(device)
    optimizer = optim.Adam(p_nn.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    
    # 获取计算域的边界
    xmin, xmax = -4, 4
    ymin, ymax = -6, 6
    
    losses = []
    pde_losses = []
    boundary_losses = []
    normalization_losses = []
    
    # 为归一化项生成300×300的高斯求积点
    nx, ny = 300, 300
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    quad_points = np.column_stack([X.ravel(), Y.ravel()])
    
    for i in tqdm(range(num_iterations)):
        # 1. 随机抽取数据点用于PDE项 (批大小 = 10,000)
        idx_data = np.random.choice(len(train_data), batch_size_data)
        x_data = torch.tensor(train_data[idx_data], dtype=torch.float32, device=device)
        x_data.requires_grad = True
        
        # 2. 随机抽取边界点 (批大小 = 4,000)
        # 创建边界点 - 随机选择边界的四条边
        n_per_side = batch_size_boundary // 4
        
        # 左边界 (x = xmin)
        left = torch.zeros(n_per_side, 2, device=device)
        left[:, 0] = xmin
        left[:, 1] = torch.rand(n_per_side, device=device) * (ymax - ymin) + ymin
        
        # 右边界 (x = xmax)
        right = torch.zeros(n_per_side, 2, device=device)
        right[:, 0] = xmax
        right[:, 1] = torch.rand(n_per_side, device=device) * (ymax - ymin) + ymin
        
        # 下边界 (y = ymin)
        bottom = torch.zeros(n_per_side, 2, device=device)
        bottom[:, 0] = torch.rand(n_per_side, device=device) * (xmax - xmin) + xmin
        bottom[:, 1] = ymin
        
        # 上边界 (y = ymax)
        top = torch.zeros(n_per_side, 2, device=device)
        top[:, 0] = torch.rand(n_per_side, device=device) * (xmax - xmin) + xmin
        top[:, 1] = ymax
        
        # 合并所有边界点
        x_boundary = torch.cat([left, right, bottom, top], dim=0)
        x_boundary.requires_grad = True
        
        # 3. 随机抽取高斯求积点用于归一化项 (批大小 = 90,000, 即 300×300)
        if len(quad_points) > batch_size_quadrature:
            idx_quad = np.random.choice(len(quad_points), batch_size_quadrature)
            x_quad = torch.tensor(quad_points[idx_quad], dtype=torch.float32, device=device)
        else:
            x_quad = torch.tensor(quad_points, dtype=torch.float32, device=device)
        
        # 计算PDE项损失
        p = p_nn(x_data)
        
        # 计算p关于x的梯度
        grad_outputs = torch.ones_like(p)
        grad_p = torch.autograd.grad(p, x_data, grad_outputs=grad_outputs, create_graph=True)[0]
        
        # 获取神经网络估计的漂移和扩散
        a_est = a_nn(x_data)
        bbt_est = b_nn(x_data)
        
        # 计算p的拉普拉斯项 (二阶导数)
        laplacian = torch.zeros_like(p)
        
        for d in range(2):
            # 计算p关于x_d的一阶导数
            p_d = grad_p[:, d].view(-1, 1)
            
            # 计算扩散项与梯度的乘积
            diff_term = torch.zeros_like(p)
            for j in range(2):
                diff_term += bbt_est[:, d, j].view(-1, 1) * grad_p[:, j].view(-1, 1)
            
            # 将扩散项与梯度的乘积再求导
            grad_diff_term = torch.autograd.grad(
                diff_term, x_data, grad_outputs=torch.ones_like(diff_term), 
                create_graph=True
            )[0][:, d].view(-1, 1)
            
            # 累加到拉普拉斯项
            laplacian += 0.5 * grad_diff_term
        
        # 计算漂移项与梯度的乘积
        drift_term = torch.zeros_like(p)
        for d in range(2):
            drift_term += a_est[:, d].view(-1, 1) * grad_p[:, d].view(-1, 1)
        
        # 计算FP方程的残差
        residual = laplacian - drift_term
        pde_loss = torch.mean(residual**2)
        
        # 计算边界条件损失 (边界上p=0)
        p_boundary = p_nn(x_boundary)
        boundary_loss = torch.mean(p_boundary**2)
        
        # 计算归一化损失 (确保概率密度积分为1)
        p_quad = p_nn(x_quad)
        area = (xmax - xmin) * (ymax - ymin)
        norm_loss = (torch.mean(p_quad) * area - 1.0)**2
        
        # 总损失 - 使用论文中的正则化参数 lambda=1, gamma=500
        loss = pde_loss + lambda_param * boundary_loss + gamma_param * norm_loss
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        pde_losses.append(pde_loss.item())
        boundary_losses.append(boundary_loss.item())
        normalization_losses.append(norm_loss.item())
        
        if (i+1) % 1000 == 0:
            print(f"Iteration {i+1}/{num_iterations}, "
                  f"Loss: {loss.item():.6f}, "
                  f"PDE Loss: {pde_loss.item():.6f}, "
                  f"Boundary Loss: {boundary_loss.item():.6f}, "
                  f"Norm Loss: {norm_loss.item():.6f}")
    
    return losses, pde_losses, boundary_losses, normalization_losses

# 4. 评估函数
def evaluate_drift_net(a_nn, grid_points, X, Y):
    """评估漂移项网络"""
    # 生成用于计算相对L2误差的10,000个高斯求积点
    eval_points, _, _ = generate_grid_points(for_eval=True)
    
    # 转换为张量
    x_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    eval_tensor = torch.tensor(eval_points, dtype=torch.float32, device=device)
    
    # 计算真实值和预测值（用于可视化）
    with torch.no_grad():
        a_true = drift(x_tensor).cpu().numpy()
        a_pred = a_nn(x_tensor).cpu().numpy()
        
        # 计算用于误差的真实值和预测值
        a_true_eval = drift(eval_tensor).cpu().numpy()
        a_pred_eval = a_nn(eval_tensor).cpu().numpy()
    
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

def evaluate_diffusion_net(b_nn, grid_points, X, Y):
    """评估扩散矩阵网络"""
    # 生成用于计算相对L2误差的10,000个高斯求积点
    eval_points, _, _ = generate_grid_points(for_eval=True)
    
    # 转换为张量
    x_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    eval_tensor = torch.tensor(eval_points, dtype=torch.float32, device=device)
    
    # 计算真实值和预测值（用于可视化）
    with torch.no_grad():
        bbt_true = diffusion_product(x_tensor).cpu().numpy()
        bbt_pred = b_nn(x_tensor).cpu().numpy()
        
        # 计算用于误差的真实值和预测值
        bbt_true_eval = diffusion_product(eval_tensor).cpu().numpy()
        bbt_pred_eval = b_nn(eval_tensor).cpu().numpy()
    
    # 提取第一个对角元素
    bbt_true_11 = bbt_true[:, 0, 0]
    bbt_pred_11 = bbt_pred[:, 0, 0]
    
    # 计算相对L2误差
    error_bbt = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            error_bbt[i, j] = np.sqrt(np.mean((bbt_true_eval[:, i, j] - bbt_pred_eval[:, i, j])**2)) / np.sqrt(np.mean(bbt_true_eval[:, i, j]**2))
    
    # 计算总体误差
    flat_true = bbt_true_eval.reshape(-1, 4)
    flat_pred = bbt_pred_eval.reshape(-1, 4)
    total_error = np.sqrt(np.mean(np.sum((flat_true - flat_pred)**2, axis=1))) / np.sqrt(np.mean(np.sum(flat_true**2, axis=1)))
    
    print(f"Diffusion relative L2 errors:")
    print(f"BB^T[0,0]: {error_bbt[0, 0]:.4e}, BB^T[0,1]: {error_bbt[0, 1]:.4e}")
    print(f"BB^T[1,0]: {error_bbt[1, 0]:.4e}, BB^T[1,1]: {error_bbt[1, 1]:.4e}")
    print(f"Total: {total_error:.4e}")
    
    # 绘制结果 - 只展示第一个对角元素
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.contourf(X, Y, bbt_true_11.reshape(X.shape), 50, cmap='viridis')
    plt.colorbar(label='True BB^T[0,0]')
    plt.title('True Diffusion BB^T[0,0]')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.subplot(132)
    plt.contourf(X, Y, bbt_pred_11.reshape(X.shape), 50, cmap='viridis')
    plt.colorbar(label='Predicted BB^T[0,0]')
    plt.title('Predicted Diffusion BB^T[0,0]')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.subplot(133)
    plt.contourf(X, Y, (bbt_true_11 - bbt_pred_11).reshape(X.shape), 50, cmap='coolwarm')
    plt.colorbar(label='Error BB^T[0,0]')
    plt.title(f'Error in BB^T[0,0] (L2 rel. error: {error_bbt[0, 0]:.4e})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.tight_layout()
    plt.savefig('results/diffusion_comparison.png')
    plt.close()
    
    return total_error

def evaluate_density_net(p_nn, grid_points, X, Y):
    """评估密度网络"""
    # 生成用于计算相对L2误差的10,000个高斯求积点
    eval_points, _, _ = generate_grid_points(for_eval=True)
    
    # 计算真实密度
    x1, x2 = grid_points[:, 0], grid_points[:, 1]
    p_true = true_density(x1, x2)
    
    # 计算用于误差评估的真实密度
    x1_eval, x2_eval = eval_points[:, 0], eval_points[:, 1]
    p_true_eval = true_density(x1_eval, x2_eval)
    
    # 计算预测密度（用于可视化）
    x_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    with torch.no_grad():
        p_pred = p_nn(x_tensor).cpu().numpy().flatten()
    
    # 计算用于误差评估的预测密度
    eval_tensor = torch.tensor(eval_points, dtype=torch.float32, device=device)
    with torch.no_grad():
        p_pred_eval = p_nn(eval_tensor).cpu().numpy().flatten()
    
    # 计算相对L2误差
    error = np.sqrt(np.mean((p_true_eval - p_pred_eval)**2)) / np.sqrt(np.mean(p_true_eval**2))
    print(f"Density relative L2 error: {error:.4e}")
    
    # 绘制结果
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.contourf(X, Y, p_true.reshape(X.shape), 50, cmap='viridis')
    plt.colorbar(label='True Density')
    plt.title('True Density')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.subplot(132)
    plt.contourf(X, Y, p_pred.reshape(X.shape), 50, cmap='viridis')
    plt.colorbar(label='Predicted Density')
    plt.title('Predicted Density')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.subplot(133)
    plt.contourf(X, Y, (p_true - p_pred).reshape(X.shape), 50, cmap='coolwarm')
    plt.colorbar(label='Error')
    plt.title(f'Error in Density (L2 rel. error: {error:.4e})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.tight_layout()
    plt.savefig('results/density_comparison.png')
    plt.close()
    
    return error

# 5. 主函数
def main():
    print("\n" + "="*50)
    print("复现论文中的Student's t-distribution实验")
    print("="*50 + "\n")
    
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    
    # 参数设置 - 按照论文要求
    dt = 0.05  # 时间步长
    N = 2*10**7  # 数据点数
    hidden_dim = 50  # 隐藏层维度
    num_blocks = 6   # 残差网络块数量（6层隐藏层）
    
    # 训练参数 - 根据论文设置
    num_iterations = 20000  # 训练迭代次数
    drift_batch_size = 10000      # 漂移项批处理大小
    diffusion_batch_size = 10000  # 扩散矩阵批处理大小
    pde_batch_size = 10000        # PDE残差项批处理大小
    boundary_batch_size = 4000    # 边界条件批处理大小
    quadrature_batch_size = 90000 # 归一化项批处理大小 (300×300)
    
    # 学习率 - 根据论文设置
    drift_lr = 1e-4        # 漂移项学习率
    diffusion_lr = 1e-4    # 扩散项学习率
    density_lr = 1e-3      # 密度网络学习率
    
    # 1. 数据准备
    data_file = "sde_data_cuda.npy"
    if os.path.exists(data_file):
        print(f"加载已有数据: {data_file}...")
        data = np.load(data_file)
        # 如果需要，可以随机采样减少数据量
        if len(data) > N:
            indices = np.random.choice(len(data), N, replace=False)
            data = data[indices]
        print(f"加载了 {len(data)} 个数据点")
    else:
        print("生成新数据...")
        data = generate_data(dt=dt, N=N)
        np.save(data_file, data)
    
    # 2. 准备评估网格
    print("生成评估网格...")
    # 可视化用的网格 (100x100)
    vis_grid_points, X, Y = generate_grid_points(nx=100, ny=100)
    # 评估误差用的网格 (10,000高斯求积点)
    eval_grid_points, _, _ = generate_grid_points(for_eval=True)
    
    # 3. 初始化模型
    print("初始化神经网络模型...")
    drift_net = DriftNet(input_dim=2, hidden_dim=hidden_dim, output_dim=2, num_blocks=num_blocks).to(device)
    diffusion_net = DiffusionNet(input_dim=2, hidden_dim=hidden_dim, output_dim=3, num_blocks=num_blocks).to(device)
    density_net = DensityNet(input_dim=2, hidden_dim=hidden_dim, output_dim=1, num_blocks=num_blocks).to(device)
    
    # 4. 训练和评估模型
    # 4.1 训练漂移项网络
    print("\n" + "-"*30)
    print("第一步: 训练漂移项网络")
    print("-"*30)
    drift_losses = train_drift_net_from_data(
        drift_net, 
        data, 
        dt=dt, 
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
    drift_error = evaluate_drift_net(drift_net, vis_grid_points, X, Y)
    print(f"漂移项网络相对L2误差: {drift_error:.4e}")
    
    # # 4.2 训练扩散矩阵网络
    # print("\n" + "-"*30)
    # print("第二步: 训练扩散矩阵网络")
    # print("-"*30)
    # diffusion_losses = train_diffusion_net(
    #     diffusion_net, 
    #     data, 
    #     num_iterations=num_iterations, 
    #     batch_size=diffusion_batch_size, 
    #     lr=diffusion_lr
    # )
    
    # # 绘制损失曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(diffusion_losses)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Diffusion Network Training Loss')
    # plt.yscale('log')
    # plt.grid(True)
    # plt.savefig('results/diffusion_training_loss.png')
    # plt.close()
    
    # # 评估扩散矩阵网络
    # print("\n评估扩散矩阵网络...")
    # diffusion_error = evaluate_diffusion_net(diffusion_net, vis_grid_points, X, Y)
    # print(f"扩散矩阵网络相对L2误差: {diffusion_error:.4e}")
    
    # # 4.3 训练密度网络
    # print("\n" + "-"*30)
    # print("第三步: 训练密度网络")
    # print("-"*30)
    # density_losses, pde_losses, boundary_losses, normalization_losses = train_density_net(
    #     density_net, 
    #     drift_net, 
    #     diffusion_net, 
    #     data, 
    #     vis_grid_points, 
    #     num_iterations=num_iterations, 
    #     batch_size_pde=pde_batch_size, 
    #     batch_size_boundary=boundary_batch_size, 
    #     batch_size_quadrature=quadrature_batch_size,
    #     lr=density_lr,
    #     lambda_param=1.0,  # 论文中的 λ₁ = 1
    #     gamma_param=500    # 论文中的 λ₂ = 500
    # )
    
    # # 绘制损失曲线
    # plt.figure(figsize=(12, 10))
    # plt.subplot(2, 2, 1)
    # plt.plot(density_losses)
    # plt.xlabel('Iteration')
    # plt.ylabel('Total Loss')
    # plt.title('Density Network Total Loss')
    # plt.yscale('log')
    # plt.grid(True)
    
    # plt.subplot(2, 2, 2)
    # plt.plot(pde_losses)
    # plt.xlabel('Iteration')
    # plt.ylabel('PDE Loss')
    # plt.title('Density Network PDE Loss')
    # plt.yscale('log')
    # plt.grid(True)
    
    # plt.subplot(2, 2, 3)
    # plt.plot(boundary_losses)
    # plt.xlabel('Iteration')
    # plt.ylabel('Boundary Loss')
    # plt.title('Density Network Boundary Loss')
    # plt.yscale('log')
    # plt.grid(True)
    
    # plt.subplot(2, 2, 4)
    # plt.plot(normalization_losses)
    # plt.xlabel('Iteration')
    # plt.ylabel('Normalization Loss')
    # plt.title('Density Network Normalization Loss')
    # plt.yscale('log')
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.savefig('results/density_training_losses.png')
    # plt.close()
    
    # # 评估密度网络
    # print("\n评估密度网络...")
    # density_error = evaluate_density_net(density_net, vis_grid_points, X, Y)
    # print(f"密度网络相对L2误差: {density_error:.4e}")
    
    # # 额外实验：使用真实系数求解密度 PDE
    # print("\n" + "-"*30)
    # print("额外实验: 使用真实系数求解密度")
    # print("-"*30)
    
    # # 创建一个新的密度网络
    # true_density_net = DensityNet(input_dim=2, hidden_dim=hidden_dim, output_dim=1, num_blocks=num_blocks).to(device)
    
    # # 使用真实系数训练密度网络
    # print("使用真实系数训练密度网络...")
    
    # # 实现一个简单的函数，直接返回真实的漂移项
    # class TrueDriftNet(nn.Module):
    #     def __init__(self):
    #         super().__init__()
        
    #     def forward(self, x):
    #         return drift(x)
    
    # # 实现一个简单的函数，直接返回真实的扩散矩阵乘积
    # class TrueDiffusionNet(nn.Module):
    #     def __init__(self):
    #         super().__init__()
        
    #     def forward(self, x):
    #         return diffusion_product(x)
    
    # true_drift_net = TrueDriftNet().to(device)
    # true_diffusion_net = TrueDiffusionNet().to(device)
    
    # # 使用真实系数训练密度网络
    # true_density_losses, true_pde_losses, true_boundary_losses, true_norm_losses = train_density_net(
    #     true_density_net, 
    #     true_drift_net, 
    #     true_diffusion_net, 
    #     data, 
    #     vis_grid_points, 
    #     num_iterations=num_iterations, 
    #     batch_size_pde=pde_batch_size, 
    #     batch_size_boundary=boundary_batch_size, 
    #     batch_size_quadrature=quadrature_batch_size,
    #     lr=density_lr,
    #     lambda_param=1.0,
    #     gamma_param=500
    # )
    
    # # 评估使用真实系数训练的密度网络
    # print("\n评估使用真实系数训练的密度网络...")
    # true_density_error = evaluate_density_net(true_density_net, vis_grid_points, X, Y)
    # print(f"使用真实系数训练的密度网络相对L2误差: {true_density_error:.4e}")
    
    # # 比较结果
    # print("\n" + "="*50)
    # print("最终结果比较")
    # print("="*50)
    # print(f"漂移项网络相对L2误差: {drift_error:.4e}")
    # print(f"扩散矩阵网络相对L2误差: {diffusion_error:.4e}")
    # print(f"使用估计系数训练的密度网络相对L2误差: {density_error:.4e}")
    # print(f"使用真实系数训练的密度网络相对L2误差: {true_density_error:.4e}")
    
    # # 可视化比较真实密度与估计密度 (使用估计系数和真实系数)
    # # 计算真实密度
    # x1, x2 = vis_grid_points[:, 0], vis_grid_points[:, 1]
    # p_true = true_density(x1, x2)
    
    # # 计算预测密度
    # x_tensor = torch.tensor(vis_grid_points, dtype=torch.float32, device=device)
    # with torch.no_grad():
    #     p_pred = density_net(x_tensor).cpu().numpy().flatten()
    #     p_true_pred = true_density_net(x_tensor).cpu().numpy().flatten()
    
    # # 绘制比较图
    # plt.figure(figsize=(18, 8))
    
    # plt.subplot(131)
    # plt.contourf(X, Y, p_true.reshape(X.shape), 50, cmap='viridis')
    # plt.colorbar(label='True Density')
    # plt.title('True Density')
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    
    # plt.subplot(132)
    # plt.contourf(X, Y, p_pred.reshape(X.shape), 50, cmap='viridis')
    # plt.colorbar(label='Predicted Density\n(Estimated Coefficients)')
    # plt.title(f'Predicted Density (L2 error: {density_error:.4e})')
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    
    # plt.subplot(133)
    # plt.contourf(X, Y, p_true_pred.reshape(X.shape), 50, cmap='viridis')
    # plt.colorbar(label='Predicted Density\n(True Coefficients)')
    # plt.title(f'Predicted Density (L2 error: {true_density_error:.4e})')
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    
    # plt.tight_layout()
    # plt.savefig('results/density_comparison_all.png')
    # plt.close()
    
    # # 5. 保存模型
    # torch.save(drift_net.state_dict(), 'results/drift_net.pth')
    # torch.save(diffusion_net.state_dict(), 'results/diffusion_net.pth')
    # torch.save(density_net.state_dict(), 'results/density_net.pth')
    # torch.save(true_density_net.state_dict(), 'results/true_density_net.pth')
    
    # print("\n实验完成。模型和结果已保存到 'results' 目录。")

# 执行主函数
if __name__ == "__main__":
    main()