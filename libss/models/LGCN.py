import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.module import Module
from torch.nn import Linear


class KHopNegativeSampler(nn.Module):
    def __init__(self, k_max=3, num_neg_samples=20, batch_size=64):
        """
        基于k-hop可达性的负样本采样器

        Args:
            k_max: 最大跳数，超过此跳数的节点作为负样本
            num_neg_samples: 每个anchor采样的负样本数量
            batch_size: batch大小
        """
        super().__init__()
        self.k_max = k_max
        self.num_neg_samples = num_neg_samples
        self.batch_size = batch_size
        self.reachable_mask = None

    def compute_reachability(self, adj: torch.Tensor) -> torch.Tensor:
        """
        计算k-hop可达性矩阵

        Returns:
            reachable: [N, N] 二值矩阵，1表示可达，0表示不可达
        """
        N = adj.size(0)
        device = adj.device

        # 二值化邻接矩阵
        adj_binary = (adj > 1e-8).float()
        adj_binary.fill_diagonal_(0)

        # 累积k-hop邻居
        reachable = adj_binary.clone()  # 1-hop
        adj_power = adj_binary.clone()

        for k in range(2, self.k_max + 1):
            adj_power = torch.mm(adj_power, adj_binary)
            reachable = reachable + adj_power  # 累积k-hop

        # 二值化：可达=1，不可达=0
        reachable = (reachable > 0).float()
        reachable.fill_diagonal_(0)  # 排除自环

        return reachable

    @torch.no_grad()
    def forward(
            self,
            adj: torch.Tensor,  # [N, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样正负样本

        Returns:
            pos_indices: [B, num_pos] 正样本索引
            neg_indices: [B, num_neg] 负样本索引
        """
        N = adj.size(0)
        device = adj.device
        assert self.batch_size <= N, "batch_size cannot exceed total node count"

        if self.reachable_mask is None or self.reachable_mask.size(0) != N:
            adj_binary = (adj > 1e-8).float()
            adj_binary.fill_diagonal_(0)

            self.reachable_mask = self.compute_reachability(adj_binary)

        reachable = self.reachable_mask
        adj_binary = (adj > 1e-8).float()
        adj_binary.fill_diagonal_(0)

        # ===== 正样本：1-hop直接邻居 =====
        pos_list = []
        neg_list = []

        for i in range(self.batch_size):
            neighbors = torch.nonzero(adj_binary[i] > 0, as_tuple=True)[0]

            if neighbors.numel() == 0:
                # 如果没有邻居，用自己（或跳过这个节点）
                print(f"WARNING: 节点{i}没有邻居")
                neighbors = torch.tensor([i], device=device)

            pos_list.append(neighbors)

            unreachable_mask = (reachable[i] == 0)  # 不可达的节点
            unreachable_mask[i] = False  # 排除自己

            neg_candidates = torch.nonzero(unreachable_mask, as_tuple=True)[0]

            if neg_candidates.numel() == 0:
                print(f"WARNING: 节点{i}没有不可达节点，图可能全连通")
                non_neighbor_mask = (adj_binary[i] == 0)
                non_neighbor_mask[i] = False
                neg_candidates = torch.nonzero(non_neighbor_mask, as_tuple=True)[0]

            if neg_candidates.numel() <= self.num_neg_samples:
                sampled_neg = neg_candidates.repeat(
                    (self.num_neg_samples + neg_candidates.numel() - 1) // neg_candidates.numel()
                )[:self.num_neg_samples]
            else:
                perm = torch.randperm(neg_candidates.numel(), device=device)[:self.num_neg_samples]
                sampled_neg = neg_candidates[perm]

            neg_list.append(sampled_neg)

        # ===== 处理正样本长度不一致的问题 =====
        # 方案1: 填充到最大长度
        max_pos_len = max(p.numel() for p in pos_list)
        pos_indices_padded = []

        for pos in pos_list:
            if pos.numel() < max_pos_len:
                # 重复填充
                pos_padded = pos.repeat((max_pos_len + pos.numel() - 1) // pos.numel())[:max_pos_len]
            else:
                # 随机采样到max_pos_len
                pos_padded = pos[torch.randperm(pos.numel(), device=device)[:max_pos_len]]
            pos_indices_padded.append(pos_padded)

        pos_indices = torch.stack(pos_indices_padded, dim=0)  # [B, max_pos_len]
        neg_indices = torch.stack(neg_list, dim=0)  # [B, num_neg_samples]

        return pos_indices, neg_indices

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):  # 默认使用bias
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Linear(in_features, out_features, bias=bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.weight.size(1))
        self.weight.weight.data.uniform_(-stdv, stdv)
        if self.weight.bias is not None:
            self.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        input: [N, in_features] 节点特征
        adj: [N, N] 邻接矩阵
        """
        support = torch.mm(adj, input)  # 聚合邻居特征
        output = self.weight(support)  # 线性变换
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class FastGraphBuilder:
    """
    快速图构建器 - 修复版，参考原代码逻辑
    """

    def __init__(self, k_nearest_neighbors=10, common_neighbors=2,batch_size=64, hop=3,
                 pruning_strategy=None, device='cuda'):
        self.k = k_nearest_neighbors
        self.common_neighbors = common_neighbors
        self.pruning_strategy = pruning_strategy
        self.device = device

        self.sampler = KHopNegativeSampler(
            k_max= hop,
            num_neg_samples= k_nearest_neighbors,
            batch_size= batch_size
        )

    def build_adjacency(self, features):
        """
        完全基于GPU的快速图构建 - 修复版
        """
        # 1. L2归一化
        features = F.normalize(features, dim=1)
        N = features.size(0)

        # 2. 计算余弦相似度
        similarity = torch.mm(features, features.t())  # [N, N]

        # 3. 获取top-k邻居
        k = min(self.k + 1, N)  # +1 是因为包含自己
        topk_sim, topk_indices = torch.topk(similarity, k, dim=1)

        # 4. 构建kNN邻接矩阵（不对称）
        adj_knn = torch.zeros(N, N, device=self.device)
        row_indices = torch.arange(N, device=self.device).unsqueeze(1).expand(N, k).reshape(-1)
        col_indices = topk_indices.reshape(-1)

        # 排除自环
        mask = row_indices != col_indices
        adj_knn[row_indices[mask], col_indices[mask]] = 1.0

        # 5. 根据剪枝策略处理
        if self.pruning_strategy == 'mutual':
            adj = adj_knn * adj_knn.t()
        elif self.pruning_strategy == 'union':
            adj = ((adj_knn + adj_knn.t() )> 0).float()
        else:
            adj = adj_knn

        # 确保无自环
        adj.fill_diagonal_(0)

        if self.common_neighbors > 0 and self.pruning_strategy != 'mutual':
            adj = self._fast_prune_by_common_neighbors(adj)

        # 7. 检查图连通性
        edge_count = (adj > 0).sum().item()
        if edge_count == 0:
            print("WARNING: 图为空，使用kNN图（无剪枝）")
            adj = (adj_knn + adj_knn.t() > 0).float()
            adj.fill_diagonal_(0)

        # 8. 归一化
        adj_normalized = self._fast_normalize(adj)
        self.adj = adj_normalized
        return adj_normalized

    def _fast_prune_by_common_neighbors(self, adj):
        """
        快速共同邻居剪枝 - 矩阵运算
        """
        common_neighbor_count = torch.mm(adj, adj.t())
        mask = (common_neighbor_count >= self.common_neighbors).float()

        # 保留原有边或共同邻居足够的边
        adj_pruned = adj * mask

        # 如果剪枝后图太稀疏，放宽条件
        if adj_pruned.sum() < adj.size(0) * 2:  # 平均度<2
            return adj  # 返回未剪枝的图

        return adj_pruned

    def _fast_normalize(self, adj):
        """
        快速对称归一化: D^(-1/2) * (A + I) * D^(-1/2)
        """
        N = adj.size(0)

        # 添加自环
        adj = adj + torch.eye(N, device=self.device)

        # 计算度
        rowsum = adj.sum(1)

        # 处理孤立节点（度为0）
        rowsum = torch.clamp(rowsum, min=1e-10)

        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

        # D^(-1/2) * A * D^(-1/2)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        return adj_normalized

    @torch.no_grad()
    def sample_structural_negatives(
            self
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sampler(
            adj=self.adj
        )

class DynamicGCN(nn.Module):
    def __init__(self, dim_in, dim_out, batch_size=64, k_neighbors=15,
                 hidden_ratio=0.5, device='cuda',
                 use_graph_cache=False,  # 调试时先关闭缓存
                 dropout=0.5,
                 Proportion_k = False):
        print(f"Building dynamic GCN with {k_neighbors} neighbors")
        super(DynamicGCN, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.device = device
        self.use_graph_cache = use_graph_cache
        self.dropout = dropout
        self.batch_size = batch_size

        self.hidden_dim = int(dim_in * hidden_ratio)

        # GCN层（添加bias）
        self.gc1 = GraphConvolution(self.dim_in, self.hidden_dim, bias=True)
        self.gc2 = GraphConvolution(self.hidden_dim, self.dim_out, bias=True)


        self.graph_builder = FastGraphBuilder(
            k_nearest_neighbors=k_neighbors,
            common_neighbors=2,
            pruning_strategy='union',
            device=device
        )

        """self.sampler = StructuralNegativeSampler(
            k_pos=k_neighbors,
            k_far_nodes=k_neighbors,
            k_neg_per_far=k_neighbors,
            batch_size=batch_size
        )"""


        # 残差连接的MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_in, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.dim_out)
        )

        # 图缓存
        self._cached_adj = None

    def forward(self, all_nodes, return_features=False):
        """
        X: [batch_size, dim_in] batch特征
        queue: [queue_size, dim_in] 或 None
        """
        N = all_nodes.size(0)


        self.adj = self.graph_builder.build_adjacency(all_nodes)

        # GCN前向传播
        Z = F.relu(self.gc1(all_nodes, self.adj))
        Z = F.dropout(Z, p=self.dropout, training=self.training)
        Z_gcn = self.gc2(Z, self.adj)

        # MLP分支（残差）
        Z_mlp = self.mlp(all_nodes)

        # 融合GCN和MLP
        Z_fused = 0.5 * Z_gcn + 0.5 * Z_mlp  # 可调整权重

        return Z_fused

    def clear_cache(self):
        """清除图缓存"""
        self._cached_adj = None

    @torch.no_grad()
    def sample_structural_negatives(
            self,
            all_hop_k_features: torch.Tensor,  # [N, D] 全部最后一跳特征
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sampler(
            hop_k_features=all_hop_k_features,
            adj=self.adj
        )

