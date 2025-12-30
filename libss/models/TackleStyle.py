import math

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale, maxabs_scale, robust_scale, scale

from core import AverageMeter


def get_color_feature(batch: torch.Tensor):
    """
    batch: [B, 3, H, W], RGB in [0,255] or [0,1]
    returns: [B, 18] (RGB 9 + HSV 9)
    """
    # 归一化到 [0,1] 并添加微小值避免零
    if batch.max() > 1.0:
        batch = batch / 255.0
    batch = batch.clamp(min=1e-8, max=1.0 - 1e-8)  # 避免极端值

    B, C, H, W = batch.shape
    num_pixels = H * W

    # --- RGB 颜色矩 ---
    pixels = batch.view(B, C, -1)  # [B,3,H*W]
    mean_rgb = pixels.mean(dim=2)

    # 计算标准差时添加微小值避免为零
    std_rgb = pixels.std(dim=2) + 1e-8

    # 三阶矩（偏度）计算，使用更稳定的方式
    centered = pixels - mean_rgb.unsqueeze(2)  # 中心化
    third_moment = (centered ** 3).mean(dim=2)  # 三阶中心矩

    # 处理三阶矩的开方：使用sign保持符号，abs确保非负
    sign = torch.sign(third_moment)
    skew_rgb = sign * (torch.abs(third_moment) ** (1 / 3))

    rgb_feats = torch.cat([mean_rgb, std_rgb, skew_rgb], dim=1)  # [B,9]

    # --- HSV 颜色矩 ---
    hsv_batch = rgb_to_hsv(batch)  # [B,3,H,W]
    hsv_batch = hsv_batch.clamp(min=1e-8, max=1.0 - 1e-8)  # HSV同样避免极端值

    pixels_hsv = hsv_batch.view(B, C, -1)
    mean_hsv = pixels_hsv.mean(dim=2)
    std_hsv = pixels_hsv.std(dim=2) + 1e-8  # 添加微小值

    centered_hsv = pixels_hsv - mean_hsv.unsqueeze(2)
    third_moment_hsv = (centered_hsv ** 3).mean(dim=2)

    sign_hsv = torch.sign(third_moment_hsv)
    skew_hsv = sign_hsv * (torch.abs(third_moment_hsv) ** (1 / 3))

    hsv_feats = torch.cat([mean_hsv, std_hsv, skew_hsv], dim=1)  # [B,9]

    # 拼接 RGB+HSV 并最终检查nan
    features = torch.cat([rgb_feats, hsv_feats], dim=1)  # [B,18]

    # 最后防御：替换可能的nan为0
    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    return features


# 确保rgb_to_hsv函数的实现正确（如果不是使用torch内置函数）
def rgb_to_hsv(rgb):
    """将RGB转换为HSV颜色空间，确保数值稳定性"""
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    max_rgb = torch.max(rgb, dim=1)[0]
    min_rgb = torch.min(rgb, dim=1)[0]
    delta = max_rgb - min_rgb + 1e-8  # 避免除以零

    h = torch.zeros_like(max_rgb)
    # 处理红色为最大值的情况
    mask = (max_rgb == r) & (delta > 1e-8)
    h[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6

    # 处理绿色为最大值的情况
    mask = (max_rgb == g) & (delta > 1e-8)
    h[mask] = (2.0 + (b[mask] - r[mask]) / delta[mask])

    # 处理蓝色为最大值的情况
    mask = (max_rgb == b) & (delta > 1e-8)
    h[mask] = (4.0 + (r[mask] - g[mask]) / delta[mask])

    h = h * 60.0  # 转换为角度 [0, 360)
    h[h < 0] += 360.0  # 确保非负
    h /= 360.0  # 归一化到 [0, 1)

    s = delta / (max_rgb + 1e-8)  # 饱和度
    v = max_rgb  # 明度

    return torch.stack([h, s, v], dim=1)


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    计算 Gram 矩阵
    Args:
        features: [B, C, H, W] 的特征图
    Returns:
        Gram 矩阵: [B, C, C]
    """
    B, C, H, W = features.size()
    # 展平成 [B, C, HW]
    features = features.view(B, C, H * W)
    # 计算通道间的内积
    G = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
    # 归一化
    G = G / (C * H * W)
    return G

class StyleDownsample(nn.Module):
    """
    对 Gram 矩阵或其他风格特征进行下采样。
    使用池化代替卷积，避免引入额外卷积权重。
    """
    def __init__(self, num_down=2, mode='avg'):
        """
        num_down: 下采样次数，每次下采样 size 缩小一半
        mode: 'avg' 平均池化, 'max' 最大池化
        """
        super().__init__()
        assert mode in ['avg', 'max'], "mode must be 'avg' or 'max'"
        self.num_down = num_down
        self.mode = mode

    def forward(self, x):
        """
        x: [B, 1, C, C]  或 [B, 1, H, W]
        输出: [B, 1, C/(2^num_down), C/(2^num_down)]
        """
        for _ in range(self.num_down):
            if self.mode == 'avg':
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
            else:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

class StyleStats(nn.Module):
    """
    计算风格特征: 每个通道的均值 + 方差
    输入:  [B, C, H, W]
    输出:  [B, 2C]
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.size()
        x = (x - x.mean(dim=[2, 3], keepdim=True)) / (x.std(dim=[2, 3], keepdim=True) + self.eps)
        # 按通道求均值 (B, C)
        mu = x.mean(dim=[2, 3])
        # 按通道求方差 (B, C)
        var = x.var(dim=[2, 3], unbiased=False) + self.eps
        std = var.sqrt()
        # 拼接均值和标准差 -> (B, 2C)
        style_vec = torch.cat([mu, std], dim=1)
        return style_vec


class FiLMFusion(nn.Module):
    def __init__(self, gram_dim, color_dim=18):
        super().__init__()
        self.gamma = nn.Linear(color_dim, gram_dim)
        self.beta  = nn.Linear(color_dim, gram_dim)

    def forward(self, feat, color_feat):
        gamma = torch.tanh(self.gamma(color_feat))  # [-1, 1]
        beta = torch.tanh(self.beta(color_feat))  # [-1, 1]
        return feat * (1 + gamma) + beta


class Router(nn.Module):
    def __init__(self, dim, channel_num, t=0.5):
        super().__init__()
        self.l1 = nn.Linear(dim, int(dim / 8))
        self.l2 = nn.Linear(int(dim / 8), channel_num)
        self.t = t
        self.activation = nn.LeakyReLU(0.1)  # 使用LeakyReLU避免死神经元

        nn.init.kaiming_normal_(self.l1.weight)
        nn.init.zeros_(self.l1.bias)
        nn.init.xavier_normal_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x_activated = self.activation(self.l1(x))

        x_activated = torch.clamp(x_activated, min=-10, max=10)
        x_normalized = F.normalize(x_activated, p=2, dim=1)

        x = self.l2(x_normalized) / max(self.t, 1e-8)

        output = F.softmax(x, dim=1)

        return output
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# ==================== 方案1: 特征分块适配器（推荐）====================
class FeatureChunkAdapter(nn.Module):
    """
    将 [B, D] 特征分块成 [B, num_chunks, D//num_chunks]
    让模型学习不同特征维度之间的关系

    适用场景：
    - 特征维度较大（如 D >= 256）
    - 希望学习特征维度间的依赖关系
    """

    def __init__(self, d_model: int, num_chunks: int = 8):
        super().__init__()
        assert d_model % num_chunks == 0, f"d_model {d_model} 必须能被 num_chunks {num_chunks} 整除"

        self.d_model = d_model
        self.num_chunks = num_chunks
        self.chunk_dim = d_model // num_chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] 输入特征
        Returns:
            [B, num_chunks, chunk_dim] 分块特征
        """
        B, D = x.shape
        # 重塑为 [B, num_chunks, chunk_dim]
        return x.view(B, self.num_chunks, self.chunk_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_chunks, chunk_dim] 分块特征
        Returns:
            [B, D] 恢复的特征
        """
        B, num_chunks, chunk_dim = x.shape
        return x.view(B, num_chunks * chunk_dim)

class ResidualCrossAttentionWithChunkAdapter(nn.Module):
    """使用特征分块适配器的交叉注意力"""

    def __init__(self, d_model: int, n_heads: int = 4,
                 num_chunks: int = 8,
                 dropout: float = 0.1,
                 batch_first: bool = True):
        super().__init__()

        self.num_chunks = num_chunks
        self.chunk_dim = d_model // num_chunks

        # 适配器
        self.adapter_x = FeatureChunkAdapter(d_model, num_chunks)
        self.adapter_y = FeatureChunkAdapter(d_model, num_chunks)

        # 注意力机制（在chunk维度上操作）
        self.attn = nn.MultiheadAttention(
            self.chunk_dim, n_heads,
            dropout=dropout,
            batch_first=batch_first
        )

        self.ln_1x = nn.LayerNorm(self.chunk_dim)
        self.ln_1y = nn.LayerNorm(self.chunk_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.chunk_dim, self.chunk_dim * 4),
            QuickGELU(),
            nn.Linear(self.chunk_dim * 4, self.chunk_dim)
        )
        self.ln_2 = nn.LayerNorm(self.chunk_dim)

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        """x, y: [B, num_chunks, chunk_dim]"""
        return self.attn(x, y, y, need_weights=False)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: [B, D] query特征
            y: [B, D] key/value特征
        Returns:
            [B, D] 增强后的特征
        """
        # 转换为序列格式
        x_seq = self.adapter_x(x)  # [B, num_chunks, chunk_dim]
        y_seq = self.adapter_y(y)  # [B, num_chunks, chunk_dim]

        # 交叉注意力
        x_seq = x_seq + self.attention(self.ln_1x(x_seq), self.ln_1y(y_seq))
        x_seq = x_seq + self.mlp(self.ln_2(x_seq))

        # 恢复原始形状
        return self.adapter_x.inverse(x_seq)  # [B, D]


import torch
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class DynamicGraphBuilder:
    """
    动态图构建器 - 适配mini-batch + Queue方案
    """

    def __init__(self, k_nearest_neighbors=10, common_neighbors=2,
                 normalization_type='normalize', device='cuda'):
        self.k = k_nearest_neighbors
        self.common_neighbors = common_neighbors
        self.normalization_type = normalization_type
        self.device = device

    def build_graph_for_batch(self, batch_features, queue_features,
                              queue_labels=None, return_sparse=False):
        """
        为当前batch和queue构建图

        Args:
            batch_features: [batch_size, feature_dim] - 当前批次特征
            queue_features: [queue_size, feature_dim] - 队列中的特征
            queue_labels: [queue_size] - 可选，用于增强同类连接
            return_sparse: 是否返回稀疏矩阵格式

        Returns:
            edge_index: [2, num_edges] - PyG格式的边索引
            edge_weight: [num_edges] - 边权重（可选）
            adj_hat: 归一化邻接矩阵
        """
        # 1. 合并特征
        batch_features = self._normalize_features(batch_features)
        queue_features = self._normalize_features(queue_features)

        all_features = np.vstack([batch_features, queue_features])
        batch_size = len(batch_features)
        total_size = len(all_features)

        # 2. 构建kNN邻接矩阵
        adj = self._construct_knn_adjacency(all_features)

        # 3. 对称化处理
        adj = self._symmetrize_adjacency(adj)

        # 4. 去除自连接
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]),
                                  shape=adj.shape)
        adj.eliminate_zeros()

        # 5. 剪枝策略 - 保留共同邻居
        adj = self._prune_by_common_neighbors(adj)

        # 6. 如果提供标签，增强同类连接
        if queue_labels is not None:
            adj = self._enhance_same_class_edges(
                adj, batch_size, queue_labels
            )

        # 7. 度归一化
        adj_hat = self._normalize_adjacency(adj)

        # 8. 转换为PyTorch Geometric格式
        edge_index, edge_weight = self._to_edge_index(adj_hat, return_weight=True)

        if return_sparse:
            return edge_index, edge_weight, adj_hat
        else:
            return edge_index, edge_weight

    def _normalize_features(self, features):
        """特征归一化"""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        if self.normalization_type == 'normalize':
            features = normalize(features)
        elif self.normalization_type == 'l2':
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        return features

    def _construct_knn_adjacency(self, features):
        """构建kNN邻接矩阵"""
        # 使用ball_tree算法构建kNN图
        nbrs = NearestNeighbors(
            n_neighbors=min(self.k + 1, len(features)),
            algorithm='ball_tree',
            n_jobs=-1  # 多线程加速
        ).fit(features)

        adj = nbrs.kneighbors_graph(features, mode='connectivity')
        return adj

    def _symmetrize_adjacency(self, adj):
        """对称化邻接矩阵"""
        adj = adj.toarray()
        adj_symmetric = np.logical_or(adj, adj.T).astype(float)
        return sp.csr_matrix(adj_symmetric)

    def _prune_by_common_neighbors(self, adj):
        """基于共同邻居的剪枝策略"""
        adj = adj.toarray()
        rows, cols = np.nonzero(adj)

        # 构建邻居字典
        neighbor_dict = {}
        for row, col in zip(rows, cols):
            if row not in neighbor_dict:
                neighbor_dict[row] = []
            neighbor_dict[row].append(col)

        # 剪枝：保留有足够共同邻居的边
        for row, col in zip(rows, cols):
            if row in neighbor_dict and col in neighbor_dict:
                common = len(set(neighbor_dict[row]) & set(neighbor_dict[col]))
                if common < self.common_neighbors:
                    adj[row][col] = 0

        adj = sp.csr_matrix(adj)
        adj.eliminate_zeros()
        return adj

    def _enhance_same_class_edges(self, adj, batch_size, queue_labels):
        """
        增强同类样本之间的连接（可选策略）

        仅对batch内样本和queue样本之间添加同类边
        """
        adj = adj.toarray()

        # 假设batch样本的标签需要单独传入，这里简化处理
        # 实际使用时可以作为参数传入batch_labels

        return sp.csr_matrix(adj)

    def _normalize_adjacency(self, adj):
        """度归一化: D^(-1/2) * A * D^(-1/2)"""
        adj = sp.coo_matrix(adj)
        adj_with_self_loop = adj + sp.eye(adj.shape[0])

        rowsum = np.array(adj_with_self_loop.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 处理孤立节点

        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        adj_normalized = (d_mat_inv_sqrt @ adj_with_self_loop @ d_mat_inv_sqrt).tocoo()

        return adj_normalized

    def _to_edge_index(self, adj_sparse, return_weight=False):
        """
        将稀疏邻接矩阵转换为PyG的edge_index格式

        Returns:
            edge_index: [2, num_edges]
            edge_weight: [num_edges] (if return_weight=True)
        """
        adj_coo = sp.coo_matrix(adj_sparse)

        edge_index = torch.tensor(
            np.vstack([adj_coo.row, adj_coo.col]),
            dtype=torch.long
        ).to(self.device)

        if return_weight:
            edge_weight = torch.tensor(
                adj_coo.data,
                dtype=torch.float
            ).to(self.device)
            return edge_index, edge_weight

        return edge_index

