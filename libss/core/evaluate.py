from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from typing import Union, Tuple
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)


def _prepare_data(outputs: Union[torch.Tensor, np.ndarray],
                  labels: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """辅助函数：将输出和标签统一为 numpy 格式并展平"""
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if outputs.ndim > 1 and outputs.shape[1] > 1:
        outputs = np.argmax(outputs, axis=1)
        # 如果是二分类概率输出（单列），则展平
    elif outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1):
        outputs = outputs.flatten()
    else:
        outputs = outputs.flatten()

    labels = labels.flatten()
    return outputs, labels


def accuracy(
        outputs: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5
) -> float:
    """计算准确率"""
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)
        labels = torch.tensor(labels)

    # 如果是多类输出且维度大于1，取预测概率最大的类别
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        preds = torch.argmax(outputs, dim=1)
    # 二分类概率输出
    elif outputs.ndim == 1 or (outputs.ndim > 1 and outputs.shape[1] == 1):
        preds = (outputs >= threshold).float()
        # 展平为一维
        preds = preds.view(-1)
    else:
        preds = outputs  # 已经是预测标签

    labels = labels.view_as(preds)

    correct = torch.sum(preds == labels).item()
    total = labels.numel()  # 总样本数

    if total == 0:
        raise ZeroDivisionError("Total number of samples is zero.")

    return correct / total


def calculate_auc(
        outputs: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray]
) -> float:
    """计算AUC分数"""
    outputs, labels = _prepare_data(outputs, labels)

    # 确保 outputs 是概率（若为 logits 需要 sigmoid）
    if outputs.ndim == 1:
        # 假设这是二分类的正类概率
        pass  # roc_auc_score 默认认为是正类概率
    else:
        # 如果是多类概率，取正类概率（如二分类，输出为 [prob_class0, prob_class1]）
        # 但通常二分类只需一列概率即可
        if outputs.ndim == 2 and outputs.shape[1] == 2:
            outputs = outputs[:, 1]  # 取正类（类别1）的概率
        elif outputs.ndim == 2 and outputs.shape[1] == 1:
            outputs = outputs.flatten()

    # 处理只有一类的情况
    if len(np.unique(labels)) == 1:
        return 0.5

    return roc_auc_score(labels, outputs)


def calculate_f1(
        outputs: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        pos_label: int = 1  # 👈 新增参数
) -> float:
    """计算F1分数"""
    outputs, labels = _prepare_data(outputs, labels)

    # 处理只有一类的情况
    if len(np.unique(labels)) == 1:
        return 0.0

    # 转换为类别标签
    preds = (outputs >= threshold).astype(int)
    return f1_score(labels, preds, pos_label=pos_label, zero_division=0)


def calculate_precision(
        outputs: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        pos_label: int = 1
) -> float:
    """计算精确率"""
    outputs, labels = _prepare_data(outputs, labels)

    # 处理只有一类的情况
    if len(np.unique(labels)) == 1:
        return 0.0

    # 转换为类别标签
    preds = (outputs >= threshold).astype(int)
    return precision_score(labels, preds, pos_label=pos_label, zero_division=0)


def calculate_recall(
        outputs: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        pos_label: int = 1  # 👈 新增参数
) -> float:
    """计算召回率"""
    outputs, labels = _prepare_data(outputs, labels)

    # 处理只有一类的情况
    if len(np.unique(labels)) == 1:
        return 0.0

    # 转换为类别标签
    preds = (outputs >= threshold).astype(int)
    return recall_score(labels, preds, pos_label=pos_label, zero_division=0)


def calculate_all_metrics(
        outputs: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
        pos_label: int = 1  # 👈 新增参数：指定哪个类别为正类
) -> dict:
    """一次性计算所有指标"""
    outputs, labels = _prepare_data(outputs, labels)

    # 确保预测是二分类标签
    preds = (outputs >= threshold).astype(int)

    # 计算各指标，使用指定的 pos_label
    acc = accuracy_score(labels, preds)

    # AUC（注意：需要确保 outputs 是概率）
    if len(np.unique(labels)) == 1:
        auc = 0.5
    else:
        try:
            # 确保 outputs 是概率，如果是 logits 需要先 sigmoid
            from scipy.special import expit  # sigmoid
            if outputs.min() < 0 or outputs.max() > 1:
                outputs = expit(outputs)  # apply sigmoid
            auc = roc_auc_score(labels, outputs)
        except Exception:
            auc = float('nan')

    f1 = f1_score(labels, preds, pos_label=pos_label, zero_division=0)
    precision = precision_score(labels, preds, pos_label=pos_label, zero_division=0)
    recall = recall_score(labels, preds, pos_label=pos_label, zero_division=0)

    return {
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }



