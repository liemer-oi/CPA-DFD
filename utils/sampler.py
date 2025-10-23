import torch
from torch.distributions import MultivariateNormal

class GaussianSampler:
    def __init__(self, device, initial_epsilon=1e-6, max_cholesky_attempts=10):
        """
        :param device: 设备 ('cuda' 或 'cpu')
        :param initial_epsilon: 初始正则化系数
        :param max_cholesky_attempts: Cholesky 分解最大尝试次数
        """
        self.device = device
        self.initial_epsilon = initial_epsilon
        self.max_attempts = max_cholesky_attempts
        self.means = torch.tensor([])      # 形状 (n_classes, dim)
        self.scale_trils = torch.tensor([]) # 形状 (n_classes, dim, dim)
        self.n_classes = 0
        self.dim = None

    def _compute_cholesky(self, cov):
        """ 处理非正定协方差矩阵，返回 Cholesky 因子 """
        dim = cov.shape[0]
        identity = torch.eye(dim)
        epsilon = self.initial_epsilon
        for _ in range(self.max_attempts):
            cov_reg = cov + epsilon * identity
            try:
                L = torch.linalg.cholesky(cov_reg)
                if not torch.any(torch.isnan(L)):
                    return L
            except RuntimeError:
                pass
            epsilon *= 10
        raise RuntimeError(f"Cholesky 分解失败，尝试 epsilon 至 {epsilon}")

    def add_class(self, mean, cov):
        """
        添加新类别
        :param mean: 均值向量 (dim,)
        :param cov: 协方差矩阵 (dim, dim)
        """
        mean = torch.as_tensor(mean)
        cov = torch.as_tensor(cov)
        
        # 初始化 dim
        if self.dim is None:
            self.dim = mean.shape[0]
        assert mean.shape == (self.dim,), f"维度不匹配: 期望 {self.dim}, 实际 {mean.shape[0]}"
        
        # 计算 Cholesky 因子
        L = self._compute_cholesky(cov)
        
        # 更新存储
        self.means = torch.cat([self.means, mean.unsqueeze(0)], dim=0)
        self.scale_trils = torch.cat([self.scale_trils, L.unsqueeze(0)], dim=0)
        self.n_classes += 1

    def sample(self, n_samples=1):
        """
        生成样本
        :param n_samples: 样本数量
        :return: 生成样本 (n_samples, dim), 类别标签 (n_samples,)
        """
        if self.n_classes == 0:
            raise RuntimeError("尚未添加任何类别")
        
        # 随机选择类别
        class_indices = torch.randint(0, self.n_classes, (n_samples,))
        
        # 批量生成样本
        selected_means = self.means[class_indices].to(self.device)        # (n_samples, dim)
        selected_L = self.scale_trils[class_indices].to(self.device)      # (n_samples, dim, dim)
        
        # 使用 MultivariateNormal 高效批量采样
        z = torch.randn(n_samples, self.dim, device=self.device)
        samples = selected_means + (selected_L @ z.unsqueeze(-1)).squeeze(-1)
        
        return samples, class_indices

    def get_params(self, class_idx):
        """ 获取指定类别的均值和协方差参数 """
        return self.means[class_idx], self.scale_trils[class_idx] @ self.scale_trils[class_idx].T
    
    def compute_mahalanobis(self, x, class_idx=None):
        """
        计算输入样本到指定类别的马氏距离平方（若未指定则计算到所有类别）
        
        :param x: 输入样本 (batch_size, dim)
        :param class_idx: 目标类别索引 (整数或 None)
        :return: 马氏距离平方 (batch_size,) 或 (batch_size, n_classes)
        """
        if self.n_classes == 0:
            raise RuntimeError("尚未添加任何类别")
        x = torch.as_tensor(x, device=self.device)
        
        # 计算差值 x - μ
        if class_idx is not None:
            # 指定单类别模式
            diff = x - self.means[class_idx].unsqueeze(0).to(self.device)    # (batch_size, dim)
            L = self.scale_trils[class_idx].unsqueeze(0).to(self.device)     # (1, dim, dim)
        else:
            # 全类别模式
            diff = x.unsqueeze(1) - self.means.unsqueeze(0).to(self.device)    # (batch_size, n_classes, dim)
            L = self.scale_trils.unsqueeze(0).to(self.device)                  # (1, n_classes, dim, dim)
        
        # 解下三角线性系统 L * y = diff^T → y = L^{-1} diff^T
        y = torch.triangular_solve(diff.unsqueeze(-1).to(L.dtype), L, upper=False).solution # 输出形状 (batch_size, [n_classes], dim, 1)
        # y = torch.triangular_solve(
        #     L,                                   # (..., dim, dim)
        #     diff.unsqueeze(-1).to(L.dtype),      # (batch_size, [n_classes], dim, 1)
        #     upper=False
        # )  # 输出形状 (batch_size, [n_classes], dim, 1)
        
        # 计算马氏距离平方: y^T y = (x-μ)^T Σ^{-1} (x-μ)
        mahalanobis_sq = (y.squeeze(-1)**2).sum(dim=-1)  # (batch_size,) 或 (batch_size, n_classes)
        return mahalanobis_sq