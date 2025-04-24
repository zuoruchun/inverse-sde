import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# 参数设置
delta_t = 0.05  # 时间步长
N = 2 * 10**7   # 总步数
d = 2           # 维度（二维 SDE）
batch_size = 10000  # 批处理大小

# 检查是否有可用的CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义 phi(x1, x2)
def phi(x):
    x1, x2 = x[:, 0], x[:, 1]
    return 1 + (2/15) * (4 * x1**2 - x1 * x2 + x2**2)

# 定义漂移 a(x)
def a(x):
    x1, x2 = x[:, 0], x[:, 1]
    a1 = -1.5 * x1 + x2
    a2 = 0.25 * x1 - 1.5 * x2
    return torch.stack([a1, a2], dim=1)

# 定义扩散 b(x)
def b(x):
    batch_size = x.shape[0]
    sqrt_phi_val = torch.sqrt(phi(x))
    
    # 创建一个批处理的2x2矩阵
    b_matrix = torch.zeros(batch_size, d, d, device=device)
    
    # 填充矩阵
    b_matrix[:, 0, 0] = sqrt_phi_val
    b_matrix[:, 0, 1] = 0
    b_matrix[:, 1, 0] = -sqrt_phi_val / 12
    b_matrix[:, 1, 1] = (torch.sqrt(torch.tensor(255.0, device=device)) / 8) * sqrt_phi_val
    
    return b_matrix

# 使用批处理和GPU加速的Euler-Maruyama方法
def simulate_sde_batched():
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
        drift = a(x_current) * delta_t
        
        # 扩散项
        diffusion_matrices = b(x_current)  # 形状: [batch_size, d, d]
        noise = torch.randn(current_batch_size, d, device=device)  # 标准正态噪声
        
        # 对每个样本应用扩散矩阵
        diffusion = torch.sqrt(torch.tensor(delta_t, device=device)) * torch.bmm(diffusion_matrices, noise.unsqueeze(-1)).squeeze(-1)
        
        # 更新状态
        x_next = x_current + drift + diffusion
        
        # 将结果转移到CPU并存储
        x_result[n+1:n+1+current_batch_size] = x_next.cpu()
        
        # 更新当前状态为新状态（用于下一个批次）
        if n + batch_size < N - 1:
            x_current = x_next.clone()
        
    return x_result

# 计时开始
start_time = time.time()

# 执行模拟
x = simulate_sde_batched()

# 计时结束
end_time = time.time()
print(f"模拟耗时: {end_time - start_time:.2f} 秒")

# 将结果转换为numpy数组进行后续分析
x_np = x.numpy()

# 检查采样点在 Omega 内的比例
Omega_x1 = [-4, 4]  # x1 的范围
Omega_x2 = [-6, 6]  # x2 的范围
in_Omega = np.sum((x_np[:, 0] >= Omega_x1[0]) & (x_np[:, 0] <= Omega_x1[1]) &
                  (x_np[:, 1] >= Omega_x2[0]) & (x_np[:, 1] <= Omega_x2[1]))
percentage = (in_Omega / N) * 100
print(f"Percentage of points in Omega: {percentage:.2f}%")

# 可视化采样点（取前 10,000 个点以加快绘图）
plt.figure(figsize=(8, 6))
plt.scatter(x_np[:10000, 0], x_np[:10000, 1], s=1, alpha=0.5)
plt.xlim(Omega_x1)
plt.ylim(Omega_x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Sampled Points from CUDA-accelerated EM Method')
plt.grid(True)
plt.savefig('sde_samples_cuda.png')
plt.show()

# 保存数据（可选）
np.save('results/sde_data_cuda.npy', x_np) 




# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import os
# from scipy.special import roots_legendre
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def phi(x):
#     x1, x2 = x[:, 0], x[:, 1]
#     return 1 + (2/15) * (4 * x1**2 - x1 * x2 + x2**2)

# # 定义漂移 a(x)
# def drift(x):
#     x1, x2 = x[:, 0], x[:, 1]
#     a1 = -1.5 * x1 + x2
#     a2 = 0.25 * x1 - 1.5 * x2
#     return torch.stack([a1, a2], dim=1)


# # 定义扩散 b(x)
# def diffusion(x, d, device):
#     batch_size = x.shape[0]
#     sqrt_phi_val = torch.sqrt(phi(x))
    
#     # 创建一个批处理的2x2矩阵
#     b_matrix = torch.zeros(batch_size, d, d, device=device)
    
#     # 填充矩阵
#     b_matrix[:, 0, 0] = sqrt_phi_val
#     b_matrix[:, 0, 1] = 0
#     b_matrix[:, 1, 0] = -sqrt_phi_val * (11 / 8)
#     b_matrix[:, 1, 1] = (torch.sqrt(torch.tensor(255.0, device=device)) / 8) * sqrt_phi_val
    
#     return b_matrix

# # 使用批处理和GPU加速的Euler-Maruyama方法
# def simulate_sde_batched(N, d, batch_size, delta_t, device):
#     # 结果数组，存储在CPU上
#     x_result = torch.zeros((N, d), dtype=torch.float64)
#     x_result[0] = torch.tensor([0.0, 0.0])  # 初始点
    
#     # 当前状态，初始为[0,0]
#     x_current = torch.zeros((batch_size, d), device=device)
    
#     # 使用批处理进行模拟
#     for n in tqdm(range(0, N-1, batch_size)):
#         # 确定当前批次的实际大小（最后一批可能小于batch_size）
#         current_batch_size = min(batch_size, N-1-n)
        
#         if current_batch_size < batch_size:
#             x_current = x_current[:current_batch_size]
        
#         # 漂移项
#         drift_term = drift(x_current) * delta_t
        
#         # 扩散项
#         diffusion_matrices = diffusion(x_current, d, device)  # 形状: [batch_size, d, d]
#         noise = torch.randn(current_batch_size, d, device=device)  # 标准正态噪声
        
#         # 对每个样本应用扩散矩阵
#         diffusion_term = torch.sqrt(torch.tensor(delta_t, device=device)) * torch.bmm(diffusion_matrices, noise.unsqueeze(-1)).squeeze(-1)
        
#         # 更新状态
#         x_next = x_current + drift_term + diffusion_term
        
#         # 将结果转移到CPU并存储
#         x_result[n+1:n+1+current_batch_size] = x_next.cpu()
        
#         # 更新当前状态为新状态（用于下一个批次）
#         if n + batch_size < N - 1:
#             x_current = x_next.clone()
        
#     return x_result

# def simulate_sde_batched1(N, d, batch_size, delta_t, num_trajectories, device):
#     """模拟多条SDE轨迹并返回平均x值"""
#     # 存储所有轨迹的x值
#     x_result = torch.zeros((N, num_trajectories, d), dtype=torch.float64)  # 所有轨迹的x值
#     x_result[0, :] = torch.tensor([0.0, 0.0])  # 初始点
    
#     x_current = torch.zeros((batch_size, num_trajectories, d), device=device)
#     x_current[:, :] = torch.tensor([0.0, 0.0])  # 初始点
    
#     for n in tqdm(range(0, N-1, batch_size)):
#         current_batch_size = min(batch_size, N-1-n)
#         if current_batch_size < batch_size:
#             x_current = x_current[:current_batch_size]
        
#         for m in range(num_trajectories):
#             drift_term = drift(x_current[:, m]) * delta_t
#             diffusion_matrices = diffusion(x_current[:, m], d, device)
#             noise = torch.randn(current_batch_size, d, device=device)
#             diffusion_term = torch.sqrt(torch.tensor(delta_t, device=device)) * torch.bmm(diffusion_matrices, noise.unsqueeze(-1)).squeeze(-1)
#             x_next = x_current[:, m] + drift_term + diffusion_term
#             x_current[:, m] = x_next
        
#         x_result[n+1:n+1+current_batch_size] = x_current.cpu()
    
#     # 计算平均x值
#     avg_x = x_result.mean(dim=1)  # 形状: (N, d)
#     return avg_x 

# dt = 0.05  # 时间步长
# d = 2   # 维度
# N = 2*10**7  # 数据点数
# hidden_dim = 50  # 隐藏层维度
# num_blocks = 6   # 残差网络块数量（6层隐藏层）
# data_batch_size = 10000  # 数据批处理大小
# # 训练参数 - 根据论文设置
# num_iterations = 20000  # 训练迭代次数
# drift_batch_size = 10000      # 漂移项批处理大小
# num_trajectories = 100

# data= simulate_sde_batched1(N, d, data_batch_size, dt, num_trajectories, device)
# data1 = simulate_sde_batched(N, d, data_batch_size, dt, device)
# np.save('results/sde_data_moreT.npy', data) 
# np.save('results/sde_data_oneT.npy', data1) 
