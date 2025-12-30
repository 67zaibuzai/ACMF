import torch.nn as nn
import torch
import torch.nn.functional as F


def js_divergence(p, q, eps=1e-6):
    """
    p, q: [B, C] softmax 概率分布
    """
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div((p+eps).log(), m, reduction='batchmean') +
                  F.kl_div((q+eps).log(), m, reduction='batchmean'))

# ------------------------------
# Focal Loss
# ------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 可以是 list 或 tensor，如 [1, 2]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N, C)  # raw logits
        targets: (N,)   # class indices
        """
        num_classes = inputs.size(1)

        # 计算 softmax 概率
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        # 提取真实类别的 log probability
        # 即 p_t = probs[i, targets[i]]
        at = None
        if self.alpha is not None:
            # alpha 是一个长度为 C 的 tensor
            alpha = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
            at = alpha.gather(0, targets)  # shape: (N,)

        # 取出每个样本真实类别的预测概率 p_t
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze()  # (N,)

        # 计算 focal loss
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * log_probs.gather(1, targets.unsqueeze(1)).squeeze()

        if self.alpha is not None:
            loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def simple_contrastive_loss(features, probs, temperature=0.5):
    """
    简化的对比损失函数，适用于二分类场景 (Z shape: B x 2)。

    正样本对：预测类别相同的样本对。
    负样本对：预测类别不同的样本对。

    Args:
        F (torch.Tensor): 原始特征，形状为 (B, dim_f)。
        probs (torch.Tensor): 模型输出，形状为 (B, 2)。可以是 logits 或概率。
        temperature (float): 温度参数。

    Returns:
        torch.Tensor: 标量损失值。
    """
    B = probs.shape[0]

    predicted_classes = torch.argmax(probs, dim=1)

    pos_mask = (predicted_classes.unsqueeze(1) == predicted_classes.unsqueeze(0))  # (B, B)
    pos_mask = pos_mask.float() * (1 - torch.eye(B, device=pos_mask.device))

    feature_norm = F.normalize(features, p=2, dim=1)  # (B, 2)
    sim_matrix = torch.mm(feature_norm, feature_norm.t()) / temperature  # (B, B)
    sim_matrix_exp = torch.exp(sim_matrix)  # (B, B)

    denominator = torch.sum(sim_matrix_exp, dim=1, keepdim=True)  # (B, 1)
    numerator = torch.sum(sim_matrix_exp * pos_mask, dim=1, keepdim=True)  # (B, 1)
    numerator = torch.clamp(numerator, min=1e-8)

    loss_per_sample = -torch.log(numerator / denominator)  # (B, 1)

    loss = loss_per_sample.mean()

    return loss


# ------------------------------
# 切片工具：避免 magic number
# ------------------------------
def split_views(probs, bs):
    """返回 (fusion, img, table) 三段"""
    fusion = probs[:bs]
    img = probs[bs:-bs]
    table = probs[-bs:]
    return fusion, img, table

class MixedConsistencyLoss(nn.Module):
    def __init__(self, num_modalities=3, warmup_epochs=10, reduction="mean", bs=64, start_out_epoch=20):
        super().__init__()
        self.num_features = num_modalities
        self.ce = nn.CrossEntropyLoss(reduction=reduction)
        self.focal = FocalLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.warmup_epochs = warmup_epochs
        self.temperature = 0.07
        self.bs = bs
        self.epsilon = 1e-6
        self.start_out_epoch = start_out_epoch

    def in_forward(self, logit_lists, labels, bs):
        """
        logit_lists: list of [3B, C] logits（未softmax）
        labels: [B]
        """
        # 标签扩展：将 [B] 扩展为 [3B]
        unlabeled_mask = labels == -1
        labels = torch.cat([labels, labels, labels], dim=0)
        labeled_mask = labels != -1

        # ===== 有标签样本：FocalLoss =====
        labeled_loss = torch.tensor(0.0, device=labels.device)
        if labeled_mask.any():
            focal_losses = [
                self.focal(logits[labeled_mask], labels[labeled_mask], gamma=2.0)
                for logits in logit_lists
            ]
            labeled_loss = sum(focal_losses) / len(focal_losses)

        # ===== 无标签样本：单向 KL 一致性 =====
        prob_lists = [F.softmax(logits, dim=-1) for logits in logit_lists]
        fusion1, img1, table1 = split_views(prob_lists[0], bs)
        fusion2, img2, table2 = split_views(prob_lists[1], bs)
        fusion3, img3, table3 = split_views(prob_lists[2], bs)

        def kl_loss(p_teacher, p_student):
            return F.kl_div(
                p_student[unlabeled_mask].log(),
                p_teacher[unlabeled_mask],
                reduction="batchmean"
            )

        img_kl = (kl_loss(img1, img2) + kl_loss(img1, img3)) / 2
        table_kl = (kl_loss(table1, table2) + kl_loss(table1, table3)) / 2
        fusion_kl = ( kl_loss(fusion1, fusion2) + kl_loss(fusion1, fusion3) + kl_loss(fusion1, img1)+ kl_loss(fusion1, table1)) / 4

        unlabeled_loss = (img_kl + table_kl + fusion_kl) / 3

        # ===== 总损失 =====
        return labeled_loss + unlabeled_loss

    def out_forward(self, prototypes_sims, logit_lists, bs, labels):
        """
        prototypes_sims: 原型相似度列表
        logit_lists: list of [3B, C] logits（未softmax）
        """
        # 原型相似度转为概率
        prototypes_probs = [F.softmax(sim, dim=-1) for sim in prototypes_sims]

        # 从融合模态生成伪标签（取前B个样本）
        pseudo_labels = prototypes_probs[0][:bs].argmax(dim=-1).detach()

        # 拆分视图（logits先转为概率）
        log_prob_lists = [F.log_softmax(logits, dim=-1) for logits in logit_lists]  # 对数概率（input）

        # 拆分对数概率分布（用于input）
        log_fusion1, log_img1, log_table1 = split_views(log_prob_lists[0], bs)
        log_fusion2, log_img2, log_table2 = split_views(log_prob_lists[1], bs)
        log_fusion3, log_img3, log_table3 = split_views(log_prob_lists[2], bs)

        prototypes_probs = [F.softmax(sim, dim=-1) for sim in prototypes_sims]

        fusion_sim1, img_sim1, table_sim1 = split_views(prototypes_probs[0], bs)
        fusion_sim2, img_sim2, table_sim2 = split_views(prototypes_probs[1], bs)
        fusion_sim3, img_sim3, table_sim3 = split_views(prototypes_probs[2], bs)

        img_loss = (self.kl(log_img1, img_sim1) +
                    self.kl(log_img2, img_sim2) +
                    self.kl(log_img3, img_sim3)) / 3

        table_loss = (self.kl(log_table1, table_sim1) +
                      self.kl(log_table2, table_sim2) +
                      self.kl(log_table3, table_sim3)) / 3

        fusion_loss = (self.kl(log_fusion1, fusion_sim1) +
                       self.kl(log_fusion2, fusion_sim2) +
                       self.kl(log_fusion3, fusion_sim3)) / 3

        modality_loss = img_loss + table_loss + fusion_loss

        logit1, logit2, logit3 = logit_lists
        f1, i1, t1 = split_views(logit1, bs)

        labels_mask = labels != -1
        unlabeled_mask = ~labels_mask
        trans_labeled_loss = self.ce(i1[labels_mask], labels[labels_mask]) + self.ce(t1[labels_mask], labels[labels_mask]) + self.ce(f1[labels_mask], labels[labels_mask])
        trans_unlabeled_loss = self.ce(i1[unlabeled_mask], pseudo_labels[unlabeled_mask]) + self.ce(t1[unlabeled_mask], pseudo_labels[unlabeled_mask])
        trans_loss = trans_labeled_loss + trans_unlabeled_loss

        contrastive_loss = self.ce(prototypes_sims[0][:bs] / self.temperature, pseudo_labels)
        return modality_loss + trans_loss + contrastive_loss

    def weight_forward(self, feature_weights, prob_lists, labels, bs):
        """
        特征权重正则化：权重应与预测误差成反比
        feature_weights: [B, num_modalities]
        probs: list of [B, C] 概率
        labels: [B]，-1 表示无标签
        """
        labels = torch.concat([labels, labels, labels], dim=0)
        labeled_mask = labels != -1
        if not labeled_mask.any():
            return 0.0

        fusion1, img1, table1 = split_views(prob_lists[0], bs)
        fusion2, img2, table2 = split_views(prob_lists[1], bs)
        fusion3, img3, table3 = split_views(prob_lists[2], bs)

        img = torch.concat([img1, img2, img3], dim=0)
        table = torch.concat([table1, table2, table3], dim=0)

        B_labeled = labeled_mask.sum()
        errs = []

        target = F.one_hot(labels[labeled_mask], num_classes=img.size(-1)).float()
        errs.append(torch.abs(img[labeled_mask] - target).mean(dim=-1, keepdim=False))
        errs.append(torch.abs(table[labeled_mask] - target).mean(dim=-1, keepdim=False))

        errs = torch.stack(errs, dim=1)

        # 计算权重目标，每行归一化
        weight_target = 1.0 / (errs + self.epsilon)
        weight_target = weight_target / weight_target.sum(dim=1, keepdim=True)

        weight_loss = F.mse_loss(feature_weights[labeled_mask], weight_target)

        return weight_loss / B_labeled

    def forward(self, classifier_logits, prototypes_sims, feature_weights, labels, table_loss, epoch=0, alpha=0.001):
        in_loss = self.in_forward(classifier_logits, labels, self.bs)
        if epoch >= self.start_out_epoch:
            out_loss = self.out_forward(prototypes_sims, classifier_logits, self.bs, labels)
        else:
            out_loss = torch.tensor(0.0, device=labels.device)
        weight_loss = self.weight_forward(feature_weights, classifier_logits, labels, self.bs)

        lambda_table = alpha
        lambda_out = 1-alpha
        lambda_in = 3.0
        all_loss = lambda_in*in_loss + lambda_out * out_loss + weight_loss + lambda_table * table_loss
        return all_loss, {
            'in_loss': in_loss,
            'out_loss': out_loss,
            'weight_loss': weight_loss,
            'table_loss': table_loss
        }


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def build_criterion(config, is_train=True):
    if is_train and config.LOSS.LOSS == 'softmax':
        criterion = MixedConsistencyLoss(bs=config.TRAIN.BATCH_SIZE_PER_GPU,start_out_epoch=config.TRAIN.START_OUT_EPOCH)
    elif config.LOSS.LOSS == 'softmax':
        criterion = FocalLoss(alpha=config.LOSS.WEIGHT)
    else:
        raise ValueError('Unkown loss {}'.format(config.LOSS.LOSS))

    return criterion