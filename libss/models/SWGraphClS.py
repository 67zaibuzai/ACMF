import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from . import DynamicGCN, FastGraphBuilder
from .TackleStyle import get_color_feature, FiLMFusion
from .register import register_model
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class EnhancedFusionNet(nn.Module):
    def __init__(self, D, max_modalities=3):
        """
        Args:
            D_in (int): 每个模态输入特征维度（假设所有模态已对齐到此维度）
            D_out (int): 输出特征维度
            max_modalities (int): 支持的最大模态数量（例如 3）
        """
        super().__init__()
        self.D = D
        self.max_modalities = max_modalities

        self.fc = nn.Linear(D * max_modalities + max_modalities, D)

        self.bn_fused = nn.BatchNorm1d(D)
        self.bn_single = nn.ModuleList([
            nn.BatchNorm1d(D) for _ in range(max_modalities)
        ])

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def _pad_features(self, feat_list, batch_size, device):
        """将 feat_list 补零至 max_modalities"""
        padded = []
        mask = torch.zeros(self.max_modalities, dtype=torch.bool, device=device)
        for i in range(self.max_modalities):
            padded.append(feat_list[i])
            mask[i] = True

        return torch.cat(padded, dim=1), mask  # [B, D_out * M], [M]

    def forward(self, feat_list):
        if not feat_list:
            raise ValueError("feat_list is empty")

        batch_size = feat_list[0].size(0)
        device = feat_list[0].device

        x_padded, active_mask = self._pad_features(feat_list, batch_size, device)  # [B, D_out * M]

        # Indicator vector: [B, M]
        indicator = active_mask.float().unsqueeze(0).expand(batch_size, -1)  # [B, M]

        # Step 2: 融合所有激活模态 → 'fused'
        x_with_ind = torch.cat([x_padded, indicator], dim=1)  # [B, D_out*M + M]
        fused = self.fc(x_with_ind)
        fused = self.bn_fused(fused)
        fused = self.relu(fused)
        fused = self.dropout(fused)

        results = []

        if len(feat_list) == 1:
            results.append(fused)
            return results

        # Step 3: 单模态增强（每个模态单独与 zero 其他模态融合）
        for i in range(len(feat_list)):
            # 构造只包含第 i 个模态的输入
            single_feats = []
            for j in range(self.max_modalities):
                if j == i:
                    single_feats.append(feat_list[i])
                else:
                    single_feats.append(torch.zeros(batch_size, self.D, device=device))
            x_single = torch.cat(single_feats, dim=1)
            # indicator for only mod i
            ind_i = torch.zeros(batch_size, self.max_modalities, device=device)
            ind_i[:, i] = 1.0
            x_single_with_ind = torch.cat([x_single, ind_i], dim=1)
            out_i = self.fc(x_single_with_ind)
            out_i = self.bn_single[i](out_i)
            out_i = self.relu(out_i)
            out_i = self.dropout(out_i)
            results.append(out_i)

        results.append(fused)

        return results

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=256):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_dim)
        )

        self.log_beta = nn.Parameter(torch.tensor(0.0))

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)  # 分割为均值和方差对数
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)

        loss_recon = F.mse_loss(x_recon, x, reduction='mean')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        loss_kl = kl_div.mean()

        min_log_beta = torch.log(torch.tensor(0.01, device=self.log_beta.device))
        clamped_log_beta = torch.clamp(self.log_beta, min=min_log_beta)
        beta = torch.exp(clamped_log_beta)

        total_loss = loss_recon + beta * loss_kl

        return z, total_loss, loss_recon, loss_kl

class FakeVAE(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=256, reduction=4, just_features=False):
        super(FakeVAE, self).__init__()
        self.just_features = just_features

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim//reduction),
            nn.BatchNorm1d(input_dim//reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(input_dim//reduction, latent_dim),
        )

    def forward(self, x):
        zero_item = torch.tensor(0.0)
        if self.just_features:
            return self.mlp(x)
        return self.mlp(x), zero_item, zero_item, zero_item

class IOEGCNCLS(nn.Module):
    def __init__(
            self,
            in_chans_img: int = 3,
            in_chans_option: int = 3,
            in_chans_semantic: int = 3,
            queue_length: int = 256,
            batch_size=64,
            spec=None,
    ):
        super(IOEGCNCLS, self).__init__()
        num_classes = spec.NUM_CLASSES
        D_out = spec.OUTPUT_DIM
        zero_init_last = spec.ZERO_INIT_LAST
        self.freeze_img = spec.FREEZE_IMG
        droprate = spec.DROP_RATE
        self.use_uni_cls = spec.USE_UNI_CLS
        self.use_fusion_cls = spec.USE_FUSION_CLS
        self.use_neighbor_cls = spec.USE_NEIGHBOR_CLS
        self.sw = spec.USE_SELF_WEIGHT
        self.use_vae = spec.USE_VAE
        self.temperature = spec.TEMPERATURE
        self.k = spec.K
        self.hop = spec.HOP
        self.batch_size = batch_size
        self.weights_temperature = spec.WEIGHT_TEMPERATURE
        self.lambda_redundancy = spec.LAMBDA_REDUNDANCY
        self.use_image = spec.USE_IMAGE
        self.use_emo = spec.USE_EMO
        self.use_ope = spec.USE_OPE
        self.num_modalities = 0
        if self.use_image:
            self.num_modalities += 1
        if self.use_ope:
            self.num_modalities += 1
        if self.use_emo:
            self.num_modalities += 1

        self.tau = 0.0
        self.INF = 1e8

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.img_backbone = nn.Sequential(
            *list(resnet.children())[:-1],  # 去掉最后的 fc 层，输出 B x 2048 x 1 x 1
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # 变成 B x 2048
        )

        self.latent_dim = D_out

        if self.use_vae:
            self.vae = VariationalAutoencoder(input_dim=2048, latent_dim=D_out)
        else:
            self.vae = FakeVAE(input_dim=2048, latent_dim=D_out)

        self.option_encoder = nn.Sequential(
            nn.Linear(in_chans_option, in_chans_option // 4),
            nn.BatchNorm1d(in_chans_option // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),

            nn.Linear(in_chans_option // 4, D_out),
            nn.BatchNorm1d(D_out),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(in_chans_semantic, in_chans_semantic // 4),
            nn.BatchNorm1d(in_chans_semantic // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),

            nn.Linear(in_chans_semantic // 4, D_out),
            nn.BatchNorm1d(D_out),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),
        )

        self.filmfusion = FiLMFusion(D_out, 18)

        if self.num_modalities == 0:
            raise ValueError("At least one modality must be enabled!")
        logging.info(f"Using {self.num_modalities} modalities.")
        self.fusion_net = EnhancedFusionNet(D_out, max_modalities=self.num_modalities)
        reduction = 1
        gnn_output_dim = D_out // reduction

        if self.use_neighbor_cls:
          self.graph_builder = FastGraphBuilder(k_nearest_neighbors=self.k, common_neighbors=2,batch_size=self.batch_size, hop=self.hop)


        self.classifier = nn.Sequential(
            nn.Linear(gnn_output_dim, gnn_output_dim),
            nn.BatchNorm1d(gnn_output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),
            nn.Linear(gnn_output_dim, num_classes)
        )

        self.init_weights(zero_init_last=zero_init_last)

    def freeze_classifier(self):
        print("============Freeze Classifier============")
        for param in self.parameters():
            param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        print("============Freeze Except Classifier============")
        for param in self.parameters():
            param.requires_grad = False

        for param in self.fusion_net.parameters():
            param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

    def freeze_except_classifier(self):
        print("============Freeze Except Classifier============")
        for param in self.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True

    def init_weights(self, zero_init_last: bool = True):
        for n, m in self.named_modules():
            # 如果模块名里含有 'resnet' 就跳过
            if 'img_backbone' in n.lower():
                continue

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_last:
            for n, m in self.named_modules():
                if 'resnet' in n.lower():
                    continue
                if hasattr(m, 'zero_init_last') and callable(m.zero_init_last):
                    m.zero_init_last()

        if self.freeze_img:
            logging.info(f"===============FREEZING IMAGE BACKBONE=================")
            for param in self.img_backbone.parameters():
                param.requires_grad = False

    def forward_features(self, img: torch.Tensor) -> torch.Tensor:
        """提取图像特征"""
        x = self.img_backbone(img)  # B x 2048
        z, vae_loss, recon_loss, kl_loss = self.vae(x)
        return z, vae_loss, recon_loss, kl_loss

    def forward_option(self, table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向选项表格，返回特征和辅助损失"""
        feature = self.option_encoder(table)
        return feature

    def forward_semantic(self, table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向语义表格，返回特征和辅助损失"""
        feature = self.semantic_encoder(table)
        return feature

    def _extract_and_fuse_features(self, views):
        feats_list = []
        total_recon_loss = torch.tensor(0.0)
        total_kl_loss = torch.tensor(0.0)

        view_idx = 0  # 用于从 views 中按顺序取数据

        # 图像模态
        if self.use_image:
            if view_idx >= len(views):
                raise ValueError("Expected image view but not provided in 'views'")
            feat, _, recon_loss, kl_loss = self.forward_features(views[view_idx])
            feats_list.append(feat)
            total_recon_loss += recon_loss
            total_kl_loss += kl_loss
            view_idx += 1

        # Option 模态
        if self.use_ope:
            if view_idx >= len(views):
                raise ValueError("Expected option view but not provided in 'views'")
            feat = self.forward_option(views[view_idx])
            feats_list.append(feat)
            view_idx += 1

        # Semantic 模态
        if self.use_emo:
            if view_idx >= len(views):
                raise ValueError("Expected semantic view but not provided in 'views'")
            feat = self.forward_semantic(views[view_idx])
            feats_list.append(feat)
            view_idx += 1

        if len(feats_list) == 0:
            raise ValueError("No modality is enabled!")

        fused_outputs = self.fusion_net(feats_list)  # List[Tensor]

        # L2 归一化
        normalized_feats = [F.normalize(feat, p=2, dim=1) for feat in fused_outputs]

        return normalized_feats, total_recon_loss, total_kl_loss

    def _compute_intra_modal_loss(self, feat1_list, feat2_list, has_augmentation):
        device = feat1_list[0].device
        num_modalities = self.num_modalities

        if not has_augmentation:
            loss_list = [self._compute_consistency_loss(feat1_list[-1], feat1_list[i])
                         for i in range(num_modalities)]
            loss = torch.mean(torch.stack(loss_list))
            return loss, torch.ones((feat1_list[-1].size(0), num_modalities), device=device)


        weights = torch.ones((feat1_list[-1].size(0), num_modalities), device=device)

        if self.sw and len(feat1_list) > 1:
            weights = self.get_weights(feat1_list[:-1], feat2_list[:-1], lambda_redundancy=self.lambda_redundancy,
                                       temperature=self.weights_temperature)

        loss_list = []

        if self.use_uni_cls:
            # 单模态增强：每个模态与融合特征的对比
            for i in range(num_modalities):
                loss1 = self._compute_consistency_loss(feat1_list[i], feat2_list[-1], weights=weights[:, i])
                loss2 = self._compute_consistency_loss(feat2_list[i], feat1_list[-1], weights=weights[:, i])
                loss_list.append((loss1 + loss2) / 2.0)

        if self.use_fusion_cls:
            # 融合特征之间的对比
            loss1 = self._compute_consistency_loss(feat1_list[-1], feat2_list[-1])
            loss2 = self._compute_consistency_loss(feat2_list[-1], feat1_list[-1])
            loss_list.append((loss1 + loss2) / 2.0)

        # 如果没有启用任何对比损失，返回零损失
        if not loss_list:
            return torch.tensor(0.0, device=device), weights

        intra_cls_loss = torch.mean(torch.stack(loss_list))
        return intra_cls_loss, weights

    def _compute_neighbor_contrast_loss(self, node_feats):
        B = self.batch_size

        self.graph_builder.build_adjacency(node_feats)
        pos_indices, neg_indices = self.graph_builder.sample_structural_negatives()

        k_pos = pos_indices.size(1)
        k_neg = neg_indices.size(1)

        node_norm = F.normalize(node_feats, dim=1)
        anchor_norm = node_norm[:B]  # [B, D]

        pos_feats = node_norm[pos_indices]  # [B, k_pos, D]
        neg_feats = node_norm[neg_indices]  # [B, k_neg, D]

        # 计算所有相似度
        pos_sim = torch.bmm(
            pos_feats, anchor_norm.unsqueeze(2)
        ).squeeze(2) / self.temperature  # [B, k_pos]

        neg_sim = torch.bmm(
            neg_feats, anchor_norm.unsqueeze(2)
        ).squeeze(2) / self.temperature  # [B, k_neg]

        # ===== 标准InfoNCE: 所有正样本同时优化 =====
        # 拼接所有样本
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, k_pos + k_neg]

        # 计算log_softmax
        log_prob = F.log_softmax(logits, dim=1)

        # 所有正样本的对数概率求和
        pos_log_prob = log_prob[:, :k_pos].sum(dim=1)  # [B]

        # 负对数似然
        loss = -pos_log_prob.mean()

        return loss

    def _apply_gnn_and_global_contrast(self, fused_feat1, queue_fusion):
        device = fused_feat1.device

        if not self.use_neighbor_cls:
            return torch.tensor(0.0, device=device)
        use_queue = queue_fusion is not None and queue_fusion.size(0) > 0

        if use_queue:
            all_nodes = torch.cat([fused_feat1, queue_fusion], dim=0)
        else:
            all_nodes = fused_feat1

        neighbor_loss = self._compute_neighbor_contrast_loss(all_nodes)
        return neighbor_loss

    def _apply_gnn_and_global_contrast_graph(self, fused_feat1, fused_feat2, queue_fusion,):
        device = fused_feat1.device

        if not self.use_neighbor_cls:
            return fused_feat1, torch.tensor(0.0, device=device)

        use_queue = queue_fusion is not None and queue_fusion.size(0) > 0

        if fused_feat2 is not None:
            if use_queue:
                fusion_gnn_v1 = self.gnn_layer(torch.cat([fused_feat1, queue_fusion], dim=0))
                fusion_gnn_v2 = self.gnn_layer(torch.cat([fused_feat2, queue_fusion], dim=0))
            else:
                fusion_gnn_v1 = self.gnn_layer(fused_feat1)
                fusion_gnn_v2 = self.gnn_layer(fused_feat2)

            node_feats_v1 = fusion_gnn_v1[:self.batch_size]
            node_feats_v2 = fusion_gnn_v2[:self.batch_size]

            gnn_v1_norm = F.normalize(node_feats_v1[:self.batch_size], dim=1)
            gnn_v2_norm = F.normalize(node_feats_v2[:self.batch_size], dim=1)

            sim_matrix = torch.matmul(gnn_v1_norm, gnn_v2_norm.T) / self.temperature
            labels = torch.arange(self.batch_size, device=device)
            augmentation_contrast_loss = F.cross_entropy(sim_matrix, labels)

            neighbor_loss_v1 = self._compute_neighbor_contrast_loss(fusion_gnn_v1, k_pos=self.k)
            neighbor_loss_v2 = self._compute_neighbor_contrast_loss(fusion_gnn_v2, k_pos=self.k)
            neighbor_contrast_loss = (neighbor_loss_v1 + neighbor_loss_v2) / 2

            # 组合损失：全局对比为主，邻居对比为辅
            extra_cls_loss = augmentation_contrast_loss + neighbor_contrast_loss

            fusion_gnn = fusion_gnn_v1[:self.batch_size]
        else:
            if use_queue:
                fusion_gnn = self.gnn_layer(torch.cat([fused_feat1, queue_fusion], dim=0))[:fused_feat1.size(0)]
            else:
                fusion_gnn = self.gnn_layer(fused_feat1)

            extra_cls_loss = torch.tensor(0.0, device=device)

        return fusion_gnn, extra_cls_loss

    def get_weights(self, z, z_aug, lambda_redundancy=0.5, temperature=0.5):
        """
        为每个样本计算模态权重

        Args:
            z: list of [B, D]
            z_aug: list of [B, D]
        Returns:
            weights: [B, n]
        """
        n = len(z)
        B = z[0].size(0)
        device = z[0].device

        scores = []

        for i in range(n):
            # 任务相关性 [B]
            task_score = F.cosine_similarity(z[i], z_aug[i], dim=-1)

            # 冗余性 [B]
            redundancy = torch.zeros(B, device=device)
            for j in range(n):
                if i != j:
                    redundancy += F.cosine_similarity(z[i], z[j], dim=-1)
            redundancy /= (n - 1)

            # 综合分数 [B]
            score_i = task_score - lambda_redundancy * redundancy
            scores.append(score_i)

        scores = torch.stack(scores, dim=1)

        # todo: 加0.7
        weights = F.softmax(scores / temperature, dim=-1) + 0.7

        return weights

    def forward(
            self,
            v1: Optional[list[torch.Tensor]] = None,
            v2: Optional[list[torch.Tensor]] = None,
            queue_fusion: Optional[torch.Tensor] = None,
            is_train = True,
            is_test = False,
            epoch = 1,
            end_epoch = 1,
    ):
        """
        前向传播
        Args:
            :param labels:  真实标签 (B,)，用于计算分类损失
            :param v1: [img1, option1, semantic1]
            :param v2: [img2, option2, semantic2]
            :param queue_sem: 语义特征队列
            :param queue_fusion: 融合特征队列
        """

        feats1, recon_loss1, kl_loss1 = self._extract_and_fuse_features(v1)

        if v2 is not None:
            feats2, recon_loss2, kl_loss2 = self._extract_and_fuse_features(v2)
            has_aug = True
        else:
            feats2, recon_loss2, kl_loss2 = None,  torch.tensor(0.0), torch.tensor(0.0)
            has_aug = False

        # 合并重建损失
        total_recon_loss = recon_loss1 + recon_loss2
        total_kl_loss = kl_loss1 + kl_loss2

        intra_cls_loss, weights = self._compute_intra_modal_loss(
            feats1, feats2, has_aug
        )

        fusion_logits = None
        extra_cls_loss = torch.tensor(0.0, device=feats1[-1].device)
        fused_feature = feats1[-1]

        if epoch > end_epoch / 5 and is_train:
            extra_cls_loss = self._apply_gnn_and_global_contrast(
                fused_feature,
                queue_fusion
            )


        if not is_train:
            extra_cls_loss = self._apply_gnn_and_global_contrast(
                fused_feature,
                queue_fusion
            )

            fusion_logits = self.classifier(fused_feature)

        return {
            "fusion_feature":  fused_feature,
            'fusion_logits':fusion_logits,
            'recon_loss': total_recon_loss,
            'kl_loss': total_kl_loss,
            'intra_cls_loss': intra_cls_loss,
            'extra_cls_loss': extra_cls_loss,
            'weights': weights.mean(dim=0),
        }

    def _compute_consistency_loss(self, z1, z2, weights=None):
        N = z1.size(0)  # 建议用 .size(0) 而不是 len()

        sim_zii = (z1 @ z1.T) / self.temperature
        sim_zjj = (z2 @ z2.T) / self.temperature
        sim_zij = (z1 @ z2.T) / self.temperature

        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)

        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)
        ], dim=0)  # [2N, 2N]

        log_sim_Z = F.log_softmax(sim_Z, dim=1)  # [2N, 2N]

        if weights is not None:
            # 关键：调整维度，按行广播
            weights_expanded = weights.repeat(2).unsqueeze(1)  # [2N, 1]
            log_sim_Z = weights_expanded * log_sim_Z  # [2N, 1] * [2N, 2N] → [2N, 2N]

        loss = -torch.diag(log_sim_Z).mean()

        return loss

    def _extract_image_features(
            self, imgs: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], float, float]:
        """提取并处理图像特征"""
        if imgs is None:
            return None, 0.0, 0.0

        img_feat, _, recon_loss, kl_loss = self.forward_features(imgs)
        color_feature = get_color_feature(imgs)
        img_feat = self.filmfusion(img_feat, color_feature)

        return img_feat, recon_loss, kl_loss

    def _extract_option_features(
            self, option: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], float]:
        """提取选项特征"""
        if option is None:
            return None, 0.0

        opt_feat = self.forward_option(option)

        return opt_feat

    def _extract_semantic_features(
            self, semantic: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], float]:
        """提取语义特征"""
        if semantic is None:
            return None, 0.0

        sem_feat = self.forward_semantic(semantic)

        return sem_feat


@register_model
def get_ioegcncls_model(config, **kwargs):
    model_spec = config.MODEL.SPEC

    ioegcncls = IOEGCNCLS(
        in_chans_img=config.DATASET.IMAGE_CHANS,
        in_chans_option=config.DATASET.OPTION_CHANNEL,
        in_chans_semantic=config.DATASET.SEMANTIC_CHANNEL,
        queue_length = config.QUEUE.QUEUE_LENGTH,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        spec=model_spec
    )

    if config.MODEL.INIT_WEIGHTS:
        ioegcncls.init_weights(
            zero_init_last=model_spec.ZERO_INIT_LAST
        )

    return ioegcncls

