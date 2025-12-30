import logging
import math

import numpy as np
import time
import torch
import torch.nn as nn
from typing import Optional, Dict,Any

import torch.distributed as dist
import torch.nn.functional as F
from tensorflow.python.saved_model.model_utils.mode_keys import is_train

from core.evaluate import calculate_all_metrics
from utils.comm import comm
from sklearn.metrics import confusion_matrix

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """统一的指标记录器"""

    def __init__(self):
        # 无监督指标
        self.unsup_loss = AverageMeter()
        self.recon_loss = AverageMeter()
        self.kl_loss = AverageMeter()
        self.cls_consistency_loss = AverageMeter()
        self.unsup_time = AverageMeter()

        # 有监督指标
        self.sup_loss = AverageMeter()
        self.acc = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.precision = AverageMeter()
        self.recall = AverageMeter()
        self.sup_time = AverageMeter()

    def update_unsupervised(self, outputs, loss, batch_size, iter_time):
        self.unsup_loss.update(loss, batch_size)
        self.recon_loss.update(outputs['recon_loss'].item(), batch_size)
        self.kl_loss.update(outputs['kl_loss'].item(), batch_size)
        self.cls_consistency_loss.update(outputs['cls_consistency_loss'].item(), batch_size)
        self.unsup_time.update(iter_time)

    def update_supervised(self, loss, metrics):
        self.sup_loss.update(loss)
        self.acc.update(metrics['acc'])
        self.auc.update(metrics['auc'])
        self.f1.update(metrics['f1'])
        self.precision.update(metrics['precision'])
        self.recall.update(metrics['recall'])

    def log_unsupervised(self, epoch, batch_idx, total_batches):
        msg = (f'[Unsup] Epoch[{epoch}][{batch_idx}/{total_batches}] '
               f'Time:{self.unsup_time.val:.3f}s '
               f'Loss:{self.unsup_loss.val:.5f}({self.unsup_loss.avg:.5f}) '
               f'Recon:{self.recon_loss.avg:.5f} KL:{self.kl_loss.avg:.5f} cls_consistency_loss:{self.cls_consistency_loss.avg:.5f}')
        logging.info(msg)

    def log_supervised(self, epoch, batch_idx, total_batches):
        msg = (f'[Sup] Epoch[{epoch}][{batch_idx}/{total_batches}] '
               f'Time:{self.sup_time.val:.3f}s '
               f'Loss:{self.sup_loss.val:.5f}({self.sup_loss.avg:.5f}) '
               f'Acc:{self.acc.avg:.4f} AUC:{self.auc.avg:.4f} F1:{self.f1.avg:.4f} Precision:{self.precision.avg} Recall:{self.recall.avg:.4f}')
        logging.info(msg)

    def log_epoch_summary(self, epoch, phase1_time, phase2_time):
        logging.info(f'\n{"=" * 80}')
        logging.info(f'Epoch {epoch} Summary:')
        logging.info(f'{"-" * 80}')
        logging.info(f'Phase 1 (Unsupervised): {phase1_time:.2f}s')
        logging.info(f'  Total Loss:       {self.unsup_loss.avg:.5f}')
        logging.info(f'  Recon Loss:       {self.recon_loss.avg:.5f}')
        logging.info(f'  KL Loss:          {self.kl_loss.avg:.5f}')
        logging.info(f'{"-" * 80}')
        logging.info(f'Phase 2 (Supervised): {phase2_time:.2f}s')
        logging.info(f'  Loss:      {self.sup_loss.avg:.5f}')
        logging.info(f'  Accuracy:  {self.acc.avg:.4f}')
        logging.info(f'  AUC:       {self.auc.avg:.4f}')
        logging.info(f'  F1 Score:  {self.f1.avg:.4f}')
        logging.info(f'  Precision: {self.precision.avg:.4f}')
        logging.info(f'  Recall:    {self.recall.avg:.4f}')
        logging.info(f'{"=" * 80}\n')

def log_to_tensorboard(writer_dict, metrics, epoch):
    """记录到TensorBoard"""
    writer = writer_dict['writer']

    # 无监督指标
    writer.add_scalar('train/unsup_loss', metrics.unsup_loss.avg, epoch)
    writer.add_scalar('train/contrastive_loss', metrics.contrastive_loss.avg, epoch)
    writer.add_scalar('train/recon_loss', metrics.recon_loss.avg, epoch)
    writer.add_scalar('train/kl_loss', metrics.kl_loss.avg, epoch)

    # 有监督指标
    writer.add_scalar('train/sup_loss', metrics.sup_loss.avg, epoch)
    writer.add_scalar('train/accuracy', metrics.acc.avg, epoch)
    writer.add_scalar('train/auc', metrics.auc.avg, epoch)
    writer.add_scalar('train/f1', metrics.f1.avg, epoch)
    writer.add_scalar('train/precision', metrics.precision.avg, epoch)
    writer.add_scalar('train/recall', metrics.recall.avg, epoch)

# ==================== 损失计算 ====================
def compute_unsupervised_loss(outputs, eps=1e-8):
    recon_loss = outputs['recon_loss']
    kl_loss = outputs['kl_loss']
    cls_consistency_loss = outputs['cls_consistency_loss']

    def adaptive_weight(aux_loss, main_loss, base_weight=1.0):
        """自适应权重计算"""
        with torch.no_grad():
            log_ratio = torch.log(torch.abs(main_loss) + eps) - torch.log(torch.abs(aux_loss) + eps)
            scale = torch.exp(torch.clamp(log_ratio, -2, 2))
            weight = base_weight * scale
        return aux_loss * weight.detach()

    total_loss = (
            recon_loss +
            adaptive_weight(kl_loss, recon_loss, 0.1) +
            adaptive_weight(cls_consistency_loss, recon_loss, 1.0)
    )

    return total_loss

def _meter_reduce(meter):
    rank = comm.local_rank
    meter_sum = torch.FloatTensor([meter.sum]).cuda(rank)
    meter_count = torch.FloatTensor([meter.count]).cuda(rank)
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


def get_prototypes(embeddings, labels, probs, th, num_classes=2, device='cuda'):
    label_mask = labels != -1

    max_probs, max_idx = torch.max(probs, dim=-1)
    max_idx[label_mask] = labels[label_mask]

    conf_mask = max_probs >= th
    mask = label_mask | conf_mask

    if mask.sum() == 0:
        prototypes = torch.zeros(num_classes, embeddings.size(1), device=device)
        class_count = torch.zeros(num_classes, 1, device=device)
        return prototypes, class_count

    hard_labels = F.one_hot(max_idx[mask], num_classes=num_classes).float()
    feat = embeddings[mask]

    class_sum = hard_labels.t() @ feat
    class_count = hard_labels.sum(dim=0, keepdim=True).t()
    class_count_safe = class_count.clone()
    class_count_safe[class_count_safe == 0] = 1e-8

    prototypes = class_sum / class_count_safe

    return prototypes, class_count

def update_queue(queue, new_embeddings):
    bs = new_embeddings.size(0)
    """更新队列，替换前 bs 个位置"""
    queue = torch.roll(queue, shifts=-bs, dims=0)
    if torch.isnan(new_embeddings).any():
        raise ValueError("NaN in embeddings before entering the queue")
    queue[:bs] = new_embeddings
    return queue

def lambda_decay(epoch, num_epochs, init_lambda=1.0, min_lambda=0.01):
    """
    计算余弦退火的lambda值，从大到小变化

    参数:
    - epoch: 当前的训练轮次
    - num_epochs: 总的训练轮次
    - init_lambda: lambda的初始值，默认为1.0
    - min_lambda: lambda的最小值，默认为0.1
    返回:
    - 基于余弦退火计算得到的当前轮次的lambda值
    """
    num_epochs = num_epochs+1
    return min_lambda + (init_lambda - min_lambda) * (1 + torch.cos(torch.tensor(epoch * torch.pi / num_epochs))) / 2

def ioe_train_one_epoch(config, train_loader, model, criterion, optimizer, epoch,
                    output_dir, tb_log_dir, queue_fusion=None,
                    writer_dict=None, scaler=None, device=None,
                    end_epoch=0):
    # -------------------------
    # 初始化指标记录器
    # -------------------------
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss_meter = AverageMeter()
    intra_cls_meter = AverageMeter()
    recon_loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    extra_cls_meter = AverageMeter()

    logging.info('=> switch to train mode')
    model.train()

    end = time.time()

    # 冻结分类层
    model.freeze_classifier()

    for i, (v1, v2, y) in enumerate(train_loader):
        data_time.update(time.time() - end)

        for j in range(len(v1)):
            v1[j] = v1[j].to(device)
            v2[j] = v2[j].to(device)
        y = y.to(device)

        bs = y.size(0)
        optimizer.zero_grad()

        # -------------------------
        # 前向传播
        # -------------------------
        out = model(v1, v2, queue_fusion, epoch = epoch, end_epoch = end_epoch)

        intra_cls_loss = out.get('intra_cls_loss')
        out_recon_loss = out.get('recon_loss', torch.tensor(0.0, device=device))
        out_kl_loss = out.get('kl_loss', torch.tensor(0.0, device=device))
        extra_cls_loss = out.get('extra_cls_loss', torch.tensor(0.0, device=device))

        fusion_feature = out['fusion_feature']

        weights = out.get('weights')

        # -------------------------
        # 更新队列
        # -------------------------
        with torch.no_grad():
            if queue_fusion is not None:
                queue_fusion = update_queue(queue_fusion, fusion_feature)

        # -------------------------
        # 总 loss 加权求和
        # -------------------------
        eps = 1e-6

        def unified_weighting(aux_loss, main_loss, base_weight=1.0):
            with torch.no_grad():
                log_ratio = torch.log(torch.abs(main_loss) + eps) - torch.log(torch.abs(aux_loss) + eps)
                scale = torch.exp(torch.clamp(log_ratio, -2, 2))  # 限制 0.1~10
                weight = base_weight * scale
            return aux_loss * weight.detach()


        total_loss = (
                intra_cls_loss +
                unified_weighting(out_recon_loss, intra_cls_loss, 1) +
                unified_weighting(out_kl_loss, intra_cls_loss, 1) +
                unified_weighting(extra_cls_loss, intra_cls_loss, 1)
        )


        # -------------------------
        # 反向传播
        # -------------------------
        total_loss.backward()

        if config.TRAIN.CLIP_GRAD_NORM > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD_NORM)

        optimizer.step()

        # -------------------------
        # 更新统计
        # -------------------------
        total_loss_meter.update(total_loss.item(), bs)
        intra_cls_meter.update(intra_cls_loss.item(), bs)
        recon_loss_meter.update(out_recon_loss.item(), bs)
        kl_loss_meter.update(out_kl_loss.item(), bs)
        extra_cls_meter.update(extra_cls_loss.item(), bs)

        batch_time.update(time.time() - end)
        end = time.time()

        # -------------------------
        # 日志打印
        # -------------------------
        if i % config.PRINT_FREQ == 0:
            if hasattr(model.classifier, 'diagnose_gcn') and i==0:
                model.classifier.diagnose_gcn(model, fusion_feature, queue_fusion)
            speed = bs / batch_time.val
            msg = (
                f"=> Epoch[{epoch}][{i}/{len(train_loader)}]: "
                f"Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                f"Speed {speed:.1f} samples/s\t"
                f"Loss {total_loss_meter.val:.5f} ({total_loss_meter.avg:.5f}) | "
                f"Intra:{intra_cls_meter.val:.3f} |"
                f"Extra:{extra_cls_meter.val:.3f} |"
                f"Recon {recon_loss_meter.avg:.4f} | "
                f"KL {kl_loss_meter.avg:.4f} | "
                f"weights: {weights}"
            )

            logging.info(msg)

        torch.cuda.synchronize()

        # -------------------------
        # 写入 TensorBoard
        # -------------------------
    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']

        writer.add_scalar('train_loss/total', total_loss_meter.avg, global_steps)
        writer.add_scalar('train_loss/intra_cls', intra_cls_meter.avg, global_steps)
        writer.add_scalar('train_loss/recon', recon_loss_meter.avg, global_steps)
        writer.add_scalar('train_loss/kl', kl_loss_meter.avg, global_steps)
        writer.add_scalar('train_loss/extra', extra_cls_meter.avg, global_steps)

        writer_dict['train_global_steps'] = global_steps + 1

    return queue_fusion



def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch,
                    output_dir, tb_log_dir, queue_fusion=None, queue_emo=None,
                    writer_dict=None, scaler=None, device=None):
    # -------------------------
    # 初始化指标记录器
    # -------------------------
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss_meter = AverageMeter()
    contrastive_loss_meter = AverageMeter()
    option_loss_meter = AverageMeter()
    semantic_loss_meter = AverageMeter()
    recon_loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    cls_consistency_loss = AverageMeter()

    logging.info('=> switch to train mode')
    model.train()

    end = time.time()

    # 冻结分类层
    model.freeze_classifier()

    for i, (v1, v2, y) in enumerate(train_loader):
        data_time.update(time.time() - end)

        for j in range(len(v1)):
            v1[j] = v1[j].to(device)
            v2[j] = v2[j].to(device)
        y = y.to(device)

        bs = y.size(0)
        optimizer.zero_grad()

        # -------------------------
        # 前向传播
        # -------------------------
        out = model(v1, v2, queue_fusion, queue_emo, epoch=epoch, all_epoch=config.TRAIN.END_EPOCH)

        out_contrastive_loss = out['contrastive_loss']
        out_option_loss = out.get('option_loss')
        out_semantic_loss = out.get('semantic_loss')
        out_recon_loss = out['recon_loss']
        out_kl_loss = out['kl_loss']
        out_cls_consistency_loss = out['cls_consistency_loss']

        fusion_feature = out['fusion_feature'].detach()
        emo_feature = out['sem_feature'].detach()

        positive_num = out['positive_num']

        # -------------------------
        # 更新队列
        # -------------------------
        with torch.no_grad():
            if queue_fusion is not None and queue_emo is not None:
                queue_fusion = update_queue(queue_fusion, fusion_feature, bs * 2)
                queue_emo = update_queue(queue_emo, emo_feature, bs * 2)

        # -------------------------
        # 总 loss 加权求和
        # -------------------------
        eps = 1e-6

        def unified_weighting(aux_loss, main_loss, base_weight=1.0):
            with torch.no_grad():
                log_ratio = torch.log(torch.abs(main_loss) + eps) - torch.log(torch.abs(aux_loss) + eps)
                scale = torch.exp(torch.clamp(log_ratio, -2, 2))  # 限制 0.1~10
                weight = base_weight * scale
            return aux_loss * weight.detach()

        total_loss = (
                out_recon_loss +
                unified_weighting(out_option_loss, out_recon_loss, 1) +
                unified_weighting(out_semantic_loss, out_recon_loss, 1) +
                unified_weighting(out_contrastive_loss, out_recon_loss, 1) +
                unified_weighting(out_kl_loss, out_recon_loss, 0.1) +
                unified_weighting(out_cls_consistency_loss, out_recon_loss, 1)
        )

        # -------------------------
        # 反向传播
        # -------------------------
        total_loss.backward()

        check_nan_after_backward(total_loss, model, optimizer)

        if config.TRAIN.CLIP_GRAD_NORM > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD_NORM)

        optimizer.step()

        # -------------------------
        # 更新统计
        # -------------------------
        total_loss_meter.update(total_loss.item(), bs)
        contrastive_loss_meter.update(out_contrastive_loss.item(), bs)
        option_loss_meter.update(out_option_loss.item(), bs)
        semantic_loss_meter.update(out_semantic_loss.item(), bs)
        recon_loss_meter.update(out_recon_loss.item(), bs)
        kl_loss_meter.update(out_kl_loss.item(), bs)
        cls_consistency_loss.update(out_cls_consistency_loss.item(), bs)

        batch_time.update(time.time() - end)
        end = time.time()

        # -------------------------
        # 日志打印
        # -------------------------
        if i % config.PRINT_FREQ == 0:
            speed = bs / batch_time.val
            msg = (
                f"=> Epoch[{epoch}][{i}/{len(train_loader)}]: "
                f"Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                f"Speed {speed:.1f} samples/s\t"
                f"Option{option_loss_meter.val:.3f} | Semantic {semantic_loss_meter.val:.3f} | "
                f"Loss {total_loss_meter.val:.5f} ({total_loss_meter.avg:.5f}) | "
                f"Contrastive {contrastive_loss_meter.avg:.4f} | "
                f"Recon {recon_loss_meter.avg:.4f} | "
                f"KL {kl_loss_meter.avg:.4f} | "
                f"CLS consistence: {cls_consistency_loss.avg:.4f} | "
                f"positive num: {positive_num}"
            )
            logging.info(msg)

        torch.cuda.synchronize()

        # -------------------------
        # 写入 TensorBoard
        # -------------------------
    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']

        writer.add_scalar('train_loss/total', total_loss_meter.avg, global_steps)
        writer.add_scalar('train_loss/contrastive', contrastive_loss_meter.avg, global_steps)
        writer.add_scalar('train_loss/option', option_loss_meter.avg, global_steps)
        writer.add_scalar('train_loss/semantic', semantic_loss_meter.avg, global_steps)
        writer.add_scalar('train_loss/recon', recon_loss_meter.avg, global_steps)
        writer.add_scalar('train_loss/kl', kl_loss_meter.avg, global_steps)
        writer.add_scalar('train_loss/cls', cls_consistency_loss.avg, global_steps)

        writer_dict['train_global_steps'] = global_steps + 1

    return queue_fusion, queue_emo

def check_nan_after_backward(loss, model, optimizer):
    """检查反向传播后的NaN梯度问题"""
    # 检测NaN/Inf问题
    nan_layers = []
    inf_layers = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            # 检查梯度
            if torch.isnan(param.grad).any():
                nan_layers.append((name, "gradient"))
            if torch.isinf(param.grad).any():
                inf_layers.append((name, "gradient"))

        # 检查参数本身
        if torch.isnan(param).any():
            nan_layers.append((name, "parameter"))
        if torch.isinf(param).any():
            inf_layers.append((name, "parameter"))

    # 检查优化器状态
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            # 找到对应的参数名
            param_name = None
            for name, param in model.named_parameters():
                if param is p:
                    param_name = name
                    break

            if param_name is None:
                continue

            state = optimizer.state[p]
            if 'exp_avg' in state and torch.isnan(state['exp_avg']).any():
                nan_layers.append((param_name, "exp_avg"))
            if 'exp_avg_sq' in state and torch.isnan(state['exp_avg_sq']).any():
                nan_layers.append((param_name, "exp_avg_sq"))

            # 报告问题
        if nan_layers or inf_layers:
            print(loss)
            # 打印有问题的层
            if nan_layers:
                print("\n包含NaN的层:")
                for layer_name, problem_type in nan_layers:
                    print(f"  - {layer_name} ({problem_type})")

                    # 打印权重信息（如果适用）
                    if '.' in layer_name and problem_type == "parameter":
                        module_name = layer_name.split('.')[0]
                        try:
                            layer = getattr(model, module_name)
                            if hasattr(layer, 'kl'):
                                print(f"    权重范围: [{layer.kl.min():.6f}, {layer.kl.max():.6f}]")
                                print(f"    权重均值: {layer.kl.mean():.6f}, 标准差: {layer.kl.std():.6f}")
                        except:
                            pass

            if inf_layers:
                print("\n包含Inf的层:")
                for layer_name, problem_type in inf_layers:
                    print(f"  - {layer_name} ({problem_type})")

            # 抛出异常
            if nan_layers:
                raise ValueError("训练过程中检测到NaN值!")
            if inf_layers:
                raise ValueError("训练过程中检测到Inf值!")

        return nan_layers, inf_layers

def ioe_finetune_one_epoch(
    config,
    train_loader,
    model,
    criterion,
    optimizer,  # 主优化器（分类层 or 全模型）
    epoch,
    output_dir,
    tb_log_dir,
    writer_dict=None,
    scaler=None,
    device=None,
    queue=None,
):
    batch_time = AverageMeter()
    acc = AverageMeter()
    auc = AverageMeter()
    f1 = AverageMeter()
    prec = AverageMeter()
    recall = AverageMeter()
    lossses = AverageMeter()

    logging.info('=> 切换到训练模式')
    model.eval()

    end = time.time()

    model.freeze_except_classifier()

    for i, (v1, v2, y) in enumerate(train_loader):
        for j in range(len(v1)):
            v1[j] = v1[j].to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(v1, queue_fusion=queue, is_train=False)

        fusion_logits = out['fusion_logits']
        metrics = calculate_all_metrics(outputs=fusion_logits, labels=y)

        loss = criterion(fusion_logits, y)
        # -------------------------
        # 反向传播
        # -------------------------
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if config.TRAIN.CLIP_GRAD_NORM > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD_NORM)

        optimizer.step()  # 更新分类层（或全模型）

        # -------------------------
        # 记录指标
        # -------------------------
        acc.update(metrics['acc'])
        auc.update(metrics['auc'])
        f1.update(metrics['f1'])
        prec.update(metrics['precision'])
        recall.update(metrics['recall'])
        lossses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = (f"=>Finetune[{i}/{len(train_loader)}]: "
                   f"loss {loss.item():.6f} "
                   f"ACC {acc.avg:.5f}\tAUC {auc.avg:.5f}\t"
                   f"F1 {f1.avg:.5f}\tPrec {prec.avg:.5f}\tRecall {recall.avg:.5f}")
            logging.info(msg)

        torch.cuda.synchronize()

    # TensorBoard 记录
    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('finetune_acc', acc.avg, global_steps)
        writer.add_scalar('finetune_auc', auc.avg, global_steps)
        writer.add_scalar('finetune_f1', f1.avg, global_steps)
        writer.add_scalar('finetune_precision', prec.avg, global_steps)
        writer.add_scalar('finetune_recall', recall.avg, global_steps)
        writer.add_scalar('finetune_loss', loss.item(), global_steps)

    return acc.avg

def dtod_finetune_one_epoch(
    config,
    train_loader,
    model,
    criterion,
    optimizer,  # 主优化器（分类层 or 全模型）
    epoch,
    output_dir,
    tb_log_dir,
    writer_dict=None,
    scaler=None,
    device=None,
    queue=None,
    end_epoch=None,
):
    batch_time = AverageMeter()
    acc = AverageMeter()
    auc = AverageMeter()
    f1 = AverageMeter()
    prec = AverageMeter()
    recall = AverageMeter()
    lossses = AverageMeter()

    logging.info('=> 切换到训练模式')
    model.eval()

    end = time.time()
    model.unfreeze_classifier()

    for i, (v1, v2, y) in enumerate(train_loader):
        for j in range(len(v1)):
            v1[j] = v1[j].to(device)
            v2[j] = v2[j].to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(v1, v2, queue_fusion=queue, is_train=False, epoch=epoch, end_epoch=end_epoch)

        fusion_feature = out.get('fusion_feature')
        intra_cls_loss = out.get('intra_cls_loss')
        out_recon_loss = out.get('recon_loss', torch.tensor(0.0, device=device))
        out_kl_loss = out.get('kl_loss', torch.tensor(0.0, device=device))
        extra_cls_loss = out.get('extra_cls_loss', torch.tensor(0.0, device=device))

        eps = 1e-6
        with torch.no_grad():
            if queue is not None:
                queue = update_queue(queue, fusion_feature)

        def unified_weighting(aux_loss, main_loss, base_weight=1.0):
            with torch.no_grad():
                log_ratio = torch.log(torch.abs(main_loss) + eps) - torch.log(torch.abs(aux_loss) + eps)
                scale = torch.exp(torch.clamp(log_ratio, -2, 2))  # 限制 0.1~10
                weight = base_weight * scale
            return aux_loss * weight.detach()

        loss_unsup = (
                intra_cls_loss +
                unified_weighting(extra_cls_loss, intra_cls_loss, 1) +
                unified_weighting(out_kl_loss, intra_cls_loss, 1) +
                unified_weighting(out_recon_loss, intra_cls_loss)
        )


        fusion_logits = out['fusion_logits']
        metrics = calculate_all_metrics(outputs=fusion_logits, labels=y)

        unsup_weight_min = 0.01
        unsup_weight_max = 0.1
        total_epochs = config.TRAIN.END_EPOCH

        unsup_weight = unsup_weight_min + (unsup_weight_max - unsup_weight_min) * (
                1 - math.cos(math.pi * epoch / total_epochs)
        ) / 2

        loss_sup = criterion(fusion_logits, y)
        loss = loss_sup + loss_unsup * unsup_weight
        # -------------------------
        # 反向传播
        # -------------------------
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if config.TRAIN.CLIP_GRAD_NORM > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD_NORM)

        optimizer.step()  # 更新分类层（或全模型）

        # -------------------------
        # 记录指标
        # -------------------------
        acc.update(metrics['acc'])
        auc.update(metrics['auc'])
        f1.update(metrics['f1'])
        prec.update(metrics['precision'])
        recall.update(metrics['recall'])
        lossses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = (f"=>Finetune[{i}/{len(train_loader)}]: "
                   f"loss {loss.item():.6f} "
                   f"ACC {acc.avg:.5f}\tAUC {auc.avg:.5f}\t"
                   f"F1 {f1.avg:.5f}\tPrec {prec.avg:.5f}\tRecall {recall.avg:.5f}")
            logging.info(msg)

        torch.cuda.synchronize()

    # TensorBoard 记录
    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('finetune_acc', acc.avg, global_steps)
        writer.add_scalar('finetune_auc', auc.avg, global_steps)
        writer.add_scalar('finetune_f1', f1.avg, global_steps)
        writer.add_scalar('finetune_precision', prec.avg, global_steps)
        writer.add_scalar('finetune_recall', recall.avg, global_steps)
        writer.add_scalar('finetune_loss', loss.item(), global_steps)

    return f1.avg, queue


def finetune_one_epoch(
    config,
    train_loader,
    model,
    criterion,
    optimizer,  # 主优化器（分类层 or 全模型）
    epoch,
    output_dir,
    tb_log_dir,
    writer_dict=None,
    scaler=None,
    device=None,
    finetune_optimizer=None,  # 可选：用于主干的优化器
):
    batch_time = AverageMeter()
    acc = AverageMeter()
    auc = AverageMeter()
    f1 = AverageMeter()
    prec = AverageMeter()
    recall = AverageMeter()
    lossses = AverageMeter()

    logging.info('=> 切换到训练模式')
    model.train()

    end = time.time()

    freeze_epoch = config.TRAIN.END_EPOCH / 2
    all_unfreeze = model.set_epoch(epoch, freeze_epoch)  # 内部控制冻结状态


    for i, (v1, v2, y) in enumerate(train_loader):
        for j in range(len(v1)):
            v1[j] = v1[j].to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(v1)

        fusion_logits = out['fusion_logits']
        emotion_logits = out['sem_logits']
        logits = 0.6 * fusion_logits + 0.4 * emotion_logits
        metrics = calculate_all_metrics(outputs=logits, labels=y)

        supervised_loss = criterion(logits, y)
        unsupervised_loss = (
            out.get('option_loss') + out['semantic_loss'] + out.get('recon_loss') + out['kl_loss']
        )
        lambda_unsupervised = 0 if epoch < freeze_epoch else (epoch - freeze_epoch)/(config.TRAIN.END_EPOCH - freeze_epoch)
        loss = supervised_loss + lambda_unsupervised * unsupervised_loss * 0.05

        # -------------------------
        # 反向传播
        # -------------------------
        optimizer.zero_grad()
        if all_unfreeze and finetune_optimizer:
            finetune_optimizer.zero_grad()

        loss.backward()

        # 梯度裁剪
        if config.TRAIN.CLIP_GRAD_NORM > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD_NORM)

        # 参数更新
        optimizer.step()  # 更新分类层（或全模型）

        if all_unfreeze and finetune_optimizer:
            finetune_optimizer.step()  # 更新主干（可选不同 LR）

        # -------------------------
        # 记录指标
        # -------------------------
        acc.update(metrics['acc'])
        auc.update(metrics['auc'])
        f1.update(metrics['f1'])
        prec.update(metrics['precision'])
        recall.update(metrics['recall'])
        lossses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = (f"=>Finetune[{i}/{len(train_loader)}]: "
                   f"loss {loss.item():.6f} "
                   f"ACC {acc.avg:.5f}\tAUC {auc.avg:.5f}\t"
                   f"F1 {f1.avg:.5f}\tPrec {prec.avg:.5f}\tRecall {recall.avg:.5f}")
            logging.info(msg)

        torch.cuda.synchronize()

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('finetune_acc', acc.avg, global_steps)
        writer.add_scalar('finetune_auc', auc.avg, global_steps)
        writer.add_scalar('finetune_f1', f1.avg, global_steps)
        writer.add_scalar('finetune_precision', prec.avg, global_steps)
        writer.add_scalar('finetune_recall', recall.avg, global_steps)
        writer.add_scalar('finetune_loss', loss.item(), global_steps)

    return acc.avg

@torch.no_grad()
def test(
        config: Any,
        test_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        output_dir: str,
        tb_log_dir: str,
        writer_dict: Optional[Dict[str, Any]] = None,
        device=None
) -> float:
    distributed = dist.is_initialized()

    batch_time = AverageMeter()
    acc = AverageMeter()
    auc = AverageMeter()
    f1 = AverageMeter()
    prec = AverageMeter()
    recall = AverageMeter()

    logging.info('=> 切换到评估模式')
    model.eval()

    end = time.time()

    for i, (v1, v2, y) in enumerate(test_loader):
        for j in range(len(v1)):
            v1[j] = v1[j].to(device)
        y = y.to(device)

        # -------------------------
        # 前向传播
        # -------------------------
        out = model(v1)

        fusion_logits = out['fusion_logits']
        emotion_logits = out['sem_logits']
        logits = 0.6 * fusion_logits + 0.4 * emotion_logits
        metrics = calculate_all_metrics(outputs=logits, labels=y)


        acc.update(metrics['acc'])
        auc.update(metrics['auc'])
        f1.update(metrics['f1'])
        prec.update(metrics['precision'])
        recall.update(metrics['recall'])
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info('=> 同步分布式计算结果...')
    comm.synchronize()

    if distributed:
        dist.barrier()
        acc = _meter_reduce(acc)
        auc = _meter_reduce(auc)
        f1 = _meter_reduce(f1)
        prec = _meter_reduce(prec)
        recall = _meter_reduce(recall)

    msg = (f"=>TEST: "
           f"ACC {acc.avg:.5f}\t"
           f"AUC {auc.avg:.5f}\t"
           f"F1 {f1.avg:.5f}\t"
           f"Prec {prec.avg:.5f}\t"
           f"Recall {recall.avg:.5f}\t")
    logging.info(msg)
    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']

        writer.add_scalar('test_acc', acc.avg, global_steps)
        writer.add_scalar('test_auc', auc.avg, global_steps)
        writer.add_scalar('test_f1', f1.avg, global_steps)
        writer.add_scalar('test_precision', prec.avg, global_steps)
        writer.add_scalar('test_recall', recall.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    test_metrics = {
        'acc': acc.avg,
        'auc': auc.avg,
        'f1': f1.avg,
        'precision': prec.avg,
        'recall': recall.avg,
    }
    return acc.avg, test_metrics

@torch.no_grad()
def ioe_test(
        test_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        device=None,
        queue=None,
) -> float:
    logging.info('=> Start 测试')
    model.eval()

    logits_list = []
    labels_list = []

    for i, (v1, v2, y) in enumerate(test_loader):
        # 将数据移动到指定设备
        v1 = [item.to(device) for item in v1]
        y = y.to(device)

        out = model(v1, queue_fusion=queue, is_train=False, is_test=True)  # 根据你的模型定义调整

        logits = out['fusion_logits']
        logits_list.append(logits[:y.size(0)])
        labels_list.append(y)

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # 计算混淆矩阵和指标
    preds = logits.argmax(dim=1)  # 取最大值对应的类别 [N, C] -> [N]
    pred_labels = preds.cpu().numpy()  # 转为 numpy
    true_labels = labels.cpu().numpy()  # 真实标签
    matrix = confusion_matrix(true_labels, pred_labels, labels=np.arange(2))
    metrics = calculate_all_metrics(outputs=logits, labels=labels)

    print("Confusion Matrix:")
    logging.info(matrix)
    print("Metrics:")
    logging.info(metrics)

    return metrics
