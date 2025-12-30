import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .TackleStyle import get_color_feature,  FiLMFusion, ResidualCrossAttentionWithChunkAdapter
from .register import register_model
from torchvision import models
from .tab_net import TabNetNoEmbeddings


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

class ITTFusionUniCLS(nn.Module):
    def __init__(
            self,
            in_chans_img: int = 3,
            in_chans_option: int = 3,
            in_chans_semantic: int = 3,
            spec=None,
    ):
        super(ITTFusion, self).__init__()
        num_classes = spec.NUM_CLASSES
        D_out = spec.OUTPUT_DIM
        zero_init_last = spec.ZERO_INIT_LAST
        self.freeze_img = spec.FREEZE_IMG
        option_N_D = spec.OPTION_N_D
        option_N_A = spec.OPTION_N_A
        semantic_N_D = spec.SEMANTIC_N_D
        semantic_N_A = spec.SEMANTIC_N_A
        droprate = spec.DROP_RATE

        self.tau = 0.6
        self.temperature = 0.1
        self._pos_alpha = 0.1
        self.lambda_emo = spec.LAMBDA_EMO

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.img_backbone = nn.Sequential(
            *list(resnet.children())[:-1],  # 去掉最后的 fc 层，输出 B x 2048 x 1 x 1
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # 变成 B x 2048
        )

        self.latent_dim = D_out  # 例如 256
        self.vae = VariationalAutoencoder(input_dim=2048, latent_dim=D_out)

        self.option_encoder = TabNetNoEmbeddings(
            input_dim=in_chans_option,
            output_dim=D_out,
            n_d=option_N_D,
            n_a=option_N_A,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            virtual_batch_size=16,
            momentum=0.02,
        )

        self.semantic_encoder = TabNetNoEmbeddings(
            input_dim=in_chans_semantic,
            output_dim=D_out,
            n_d=semantic_N_D,
            n_a=semantic_N_A,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            virtual_batch_size=16,
            momentum=0.02,
        )

        self.sem_attn_net = ResidualCrossAttentionWithChunkAdapter(d_model=D_out, dropout=droprate)

        self.filmfusion = FiLMFusion(D_out, 18)

        self.fusion_net =nn.Sequential(
            nn.Linear(D_out*2, D_out),
            nn.BatchNorm1d(D_out),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(D_out, D_out // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(D_out // 8, num_classes)
        )

        self.init_weights(zero_init_last=zero_init_last)

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

    def freeze_all_except_classifier(self):
        logging.info("---解冻分类层---")
        for param in self.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def _progressive_unfreeze_backbone(self, step):
        fusion_net = [
            'fusion_net', 'sem_attn_net', 'filmfusion',
            'classifier', 'vae'
        ]
        if step < 5:
            layers = fusion_net
        elif step < 10:
            layers = fusion_net + ['7']
        elif step < 15:
            layers = fusion_net + ['5', '6', '7']
        else:
            self.unfreeze_all()
            return

        for name, param in self.named_parameters():
            param.requires_grad = any(
                f".{layer}." in name or name.startswith(layer)
                for layer in layers
            )
        if step in [0, 10, 15]:
            print(f"[Step {step}] Unfreezing layers: {layers}")

    def unfreeze_all(self):
        """解冻所有参数，允许整个模型进行梯度更新"""
        logging.info("---解冻所有层---")
        for param in self.parameters():
            param.requires_grad = True

    def set_epoch(self, step, freeze_step):
        if step < freeze_step:
            self.freeze_all_except_classifier()
        else:
            self._progressive_unfreeze_backbone(step-freeze_step)
        if step - freeze_step < 15:
            return False
        else:
            return True

    def freeze_classifier(self):
        print("============Freeze Classifier============")
        for param in self.classifier.parameters():
            param.requires_grad = False

    def forward_features(self, img: torch.Tensor) -> torch.Tensor:
        """提取图像特征"""
        x = self.img_backbone(img)  # B x 2048
        z, vae_loss, recon_loss, kl_loss = self.vae(x)
        return z, vae_loss, recon_loss, kl_loss

    def forward_option(self, table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向选项表格，返回特征和辅助损失"""
        feature, mask_loss = self.option_encoder(table)
        return feature, mask_loss

    def forward_semantic(self, table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向语义表格，返回特征和辅助损失"""
        feature, mask_loss = self.semantic_encoder(table)
        return feature, mask_loss

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """分类头前向"""
        return self.classifier(x)

    def emotionguideadaptiveCL(
            self,
            fused_feature,
            emo_feature,
            queue_fusion,
            queue_emo,
            epoch,
            all_epoch
    ):
        B, D = fused_feature.size()
        N = queue_fusion.size(0)

        # ==================== 语义相似度计算 ====================
        sim = torch.mm(fused_feature, queue_fusion.t())  # [B, N]

        # ==================== 情感相似度计算 ====================
        emo_sim = torch.mm(emo_feature, queue_emo.t())  # [B, N]

        # ==================== 动态温度参数（从冷到热）====================
        progress = epoch / all_epoch
        temperature = self.temperature * (1.0 - 0.3 * progress)

        # ==================== 情感权重自适应调整 ====================
        lambda_emo = self.lambda_emo * (1.0 / (1.0 + math.exp(-10 * (progress - 0.5))))
        sim_weighted = sim + lambda_emo * emo_sim

        # ==================== 7. 自适应正样本选择 ====================
        # ==================== 自适应阈值计算 ====================
        # 分别为语义和情感相似度计算自适应阈值
        top_k_ratio = 0.15
        k = max(1, int(N * top_k_ratio))

        # 语义相似度的自适应阈值
        topk_sim, _ = torch.topk(sim, k, dim=1)  # [B, k]
        adaptive_tau_sim = topk_sim[:, -1].unsqueeze(1)  # [B, 1]
        effective_tau_sim = 0.7 * self.tau + 0.3 * adaptive_tau_sim.mean().item()

        # 情感相似度的自适应阈值（通常设置较低，因为情感特征可能更稀疏）
        if self.lambda_emo != 0:
            topk_emo, _ = torch.topk(emo_sim, k, dim=1)  # [B, k]
            adaptive_tau_emo = topk_emo[:, -1].unsqueeze(1)  # [B, 1]
            effective_tau_emo = 0.7 * self.tau + 0.3 * adaptive_tau_emo.mean().item()

            w_pos = (sim > effective_tau_sim) & (emo_sim > effective_tau_emo)  # [B, N]

            # ==================== 8. 硬负样本挖掘 ====================
            # 选择与正样本相似但不是正样本的作为硬负样本
            hard_neg_type1 = (
                    (sim > (effective_tau_sim - 0.05)) &  # 语义相似度接近阈值
                    (emo_sim < (effective_tau_emo - 0.05)) &  # 情感相似度明显低于阈值
                    (~w_pos)
            )
            hard_neg_type2 = ((sim_weighted > (effective_tau_sim - 0.1)) & (sim_weighted < (effective_tau_sim - 0.05)) & (~w_pos))

            hard_neg_mask = hard_neg_type1 | hard_neg_type2
        else:
            w_pos = (sim > effective_tau_sim)
            hard_neg_mask = ((sim_weighted > (effective_tau_sim - 0.1)) & (sim_weighted < (effective_tau_sim - 0.05)) & (~w_pos))

        # ==================== 对比学习损失计算 ====================
        logits = sim_weighted / temperature

        # 数值稳定性：减去最大值
        logits_max = logits.max(dim=1, keepdim=True)[0].detach()
        logits_stable = logits - logits_max

        # 计算exp
        exp_logits = torch.exp(logits_stable)

        # 应用硬负样本权重
        weighted_exp = exp_logits.clone()
        weighted_exp[hard_neg_mask] *= 1.5

        # 分子：正样本的加权和
        numerator = (w_pos.float() * exp_logits).sum(dim=1)

        # 分母：所有样本（包括硬负样本加权）
        denominator = weighted_exp.sum(dim=1)

        # 避免log(0)
        loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8)).mean()

        pos_ratio = w_pos.float().mean().item()

        if pos_ratio < 0.05:
            tau = self.tau - 0.02  # 小幅下降
        elif pos_ratio > 0.2:
            tau = self.tau + 0.05  # 大幅上升
        else:
            tau = self.tau * 0.99 + 0.001 * torch.randn(1).item()  # 微调一点点随机扰动

        self.tau = min(tau, 0.99)

        return loss, [self.tau, pos_ratio, torch.max(sim_weighted).item(), torch.min(sim_weighted).item(), torch.max(sim).item(), torch.min(sim).item(), torch.max(emo_sim).item(), torch.min(emo_sim).item()]

    def _process_single_view(self, view_data: Optional[list[torch.Tensor]], view_name: str) -> dict:
        """处理单个视图的数据"""
        if view_data is None:
            return self._create_empty_view_results()

        img, opt, sem = view_data[0], view_data[1], view_data[2]

        # 提取特征
        img_feat, recon_loss, kl_loss = self._extract_image_features(img)
        opt_feat, mask_loss_opt = self._extract_option_features(opt)
        sem_feat, mask_loss_sem = self._extract_semantic_features(sem)

        # 融合特征
        fused_feat = self.fusion_net(torch.cat([img_feat, opt_feat], dim=1))
        fused_feat = F.normalize(fused_feat, p=2, dim=1)

        # 重建损失
        fusion_recon_loss = (F.mse_loss(fused_feat, img_feat, reduction='mean') +
                             F.mse_loss(fused_feat, opt_feat, reduction='mean'))

        # 语义特征增强
        sem_feat = self.sem_attn_net(sem_feat, fused_feat)
        sem_feat = F.normalize(sem_feat, p=2, dim=1)

        # 分类预测
        fusion_logits = self.classifier(fused_feat)
        sem_logits = self.classifier(sem_feat)

        # 视图内一致性损失
        cls_consistency_loss = self._compute_consistency_loss(fused_feat, sem_feat)

        return {
            "fused_feature": fused_feat,
            "sem_feature": sem_feat,
            "fusion_logits": fusion_logits,
            "sem_logits": sem_logits,
            "recon_loss": recon_loss + fusion_recon_loss,
            "kl_loss": kl_loss,
            "option_loss": mask_loss_opt,
            "semantic_loss": mask_loss_sem,
            "cls_consistency_loss": cls_consistency_loss,
            "view_name": view_name
        }

    def forward(
            self,
            v1: Optional[list[torch.Tensor]] = None,
            v2: Optional[list[torch.Tensor]] = None,
            queue_fusion: Optional[torch.Tensor] = None,
            queue_sem: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            epoch=0,
            all_epoch=1
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
        # 图像特征
        img1, opt1, sem1 = v1[0], v1[1], v1[2]

        # 处理第一个增强版本
        img_feat1, recon_loss1, kl_loss1 = self._extract_image_features(img1)
        opt_feat1, mask_loss_opt1 = self._extract_option_features(opt1)
        sem_feat1, mask_loss_sem1 = self._extract_semantic_features(sem1)

        # 图像与表格融合（两个版本）
        fused_feat1 = self.fusion_net(torch.cat([img_feat1, opt_feat1], dim=1))
        fused_feat1 = F.normalize(fused_feat1, p=2, dim=1)

        # 情感特征增强（两个版本）
        sem_feat1 = self.sem_attn_net(sem_feat1, fused_feat1)
        sem_feat1 = F.normalize(sem_feat1, p=2, dim=1)

        fusion_logits1 = self.classifier(fused_feat1)
        sem_logits1 = self.classifier(sem_feat1)

        recon_loss2, kl_loss2, mask_loss_opt2, mask_loss_sem2 = 0.0, 0.0, 0.0, 0.0
        fused_feat2 = None
        sem_feat2 = None
        if v2 is not None:
            img2, opt2, sem2 = v2[0], v2[1], v2[2]

            # 处理第二个增强版本
            img_feat2, recon_loss2, kl_loss2 = self._extract_image_features(img2)
            opt_feat2, mask_loss_opt2 = self._extract_option_features(opt2)
            sem_feat2, mask_loss_sem2 = self._extract_semantic_features(sem2)

            fused_feat2 = self.fusion_net(torch.cat([img_feat2, opt_feat2], dim=1))
            fused_feat2 = F.normalize(fused_feat2, p=2, dim=1)

            sem_feat2 = self.sem_attn_net(sem_feat2, fused_feat2)
            sem_feat2 = F.normalize(sem_feat2, p=2, dim=1)

            zs1 = v1.append(fused_feat1)
            zs2 = v2.append(fused_feat2)
            loss  = []
            for i in range(4):
                loss1 = self._compute_consistency_loss(zs1[i], zs2[-1])
                loss2 = self._compute_consistency_loss(zs2[i], zs1[-1])
                loss.append((loss2 + loss1) / 2.)
            cls_consistency_loss = torch.mean(torch.stack(loss))
        else:
            cls_consistency_loss = self._compute_consistency_loss(fused_feat1, sem_feat1) + F.mse_loss(fused_feat1, img_feat1, reduction='mean') + F.mse_loss(fused_feat1, opt_feat1,reduction='mean')


            # 对比学习 - 使用两个增强版本的特征
        contrastive_loss = torch.tensor(0.0, device=fused_feat1.device)
        positive_num = 0

        if queue_fusion is not None and queue_sem is not None:
            queue_fusion = queue_fusion.to(fused_feat1.device)
            queue_sem = queue_sem.to(fused_feat1.device)

            contrastive_loss, positive_num = self.emotionguideadaptiveCL(
                fused_feat1, sem_feat1, queue_fusion, queue_sem, epoch, all_epoch
            )

        # 总重建和KL损失
        total_recon_loss = recon_loss1 + recon_loss2
        total_kl_loss = kl_loss1 + kl_loss2

        return {
            "fusion_feature": torch.cat([fused_feat1, fused_feat2], dim=0) if fused_feat2 is not None else fused_feat1,
            "sem_feature": torch.cat([sem_feat1, sem_feat2], dim=0) if sem_feat2 is not None else sem_feat1,
            "fusion_logits": fusion_logits1,
            "sem_logits": sem_logits1,
            "option_loss": mask_loss_opt1 + mask_loss_opt2,
            'semantic_loss': mask_loss_sem1 + mask_loss_sem2,
            'recon_loss': total_recon_loss,
            'kl_loss': total_kl_loss,
            'contrastive_loss': contrastive_loss,
            'cls_consistency_loss': cls_consistency_loss,
            'positive_num': positive_num,
        }

    def _compute_consistency_loss(self, z1, z2): # InfoNCE Loss
        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = F.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()

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
        img_feat = F.normalize(img_feat, p=2, dim=1)

        return img_feat, recon_loss, kl_loss

    def _extract_option_features(
            self, option: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], float]:
        """提取选项特征"""
        if option is None:
            return None, 0.0

        opt_feat, mask_loss = self.forward_option(option)
        opt_feat = F.normalize(opt_feat, p=2, dim=1)

        return opt_feat, mask_loss

    def _extract_semantic_features(
            self, semantic: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], float]:
        """提取语义特征"""
        if semantic is None:
            return None, 0.0

        sem_feat, mask_loss = self.forward_semantic(semantic)
        sem_feat = F.normalize(sem_feat, p=2, dim=1)

        return sem_feat, mask_loss

@register_model
def get_ittfusion_model(config, **kwargs):
    model_spec = config.MODEL.SPEC

    itf = ITTFusion(
        in_chans_img=config.DATASET.IMAGE_CHANS,
        in_chans_option=config.DATASET.OPTION_CHANNEL,
        in_chans_semantic=config.DATASET.SEMANTIC_CHANNEL,
        spec=model_spec
    )

    if config.MODEL.INIT_WEIGHTS:
        itf.init_weights(
            zero_init_last=model_spec.ZERO_INIT_LAST
        )

    return itf

