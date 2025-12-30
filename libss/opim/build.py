from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

from timm.optim import create_optimizer

def _is_depthwise(m):
    return (
        isinstance(m, nn.Conv2d)
        and m.groups == m.in_channels
        and m.groups == m.out_channels
    )


def exclude_classifier_params(model):
    """
    排除分类层的参数，返回 [(name, param)] 列表
    """
    classifier_name = 'classifier'
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if classifier_name not in name:
            params.append((name, param))
    return params


def only_classifier_params(model):
    """
    只选择分类层的参数，返回 [(name, param)] 列表
    """
    classifier_name = 'classifier'
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if classifier_name in name:
            params.append((name, param))
    return params


import torch
import torch.nn as nn
from typing import List, Union, Tuple, Any


def set_wd(
        cfg: Any,
        model: nn.Module,
        param_groups_or_named_params: Union[List[Tuple[str, torch.nn.Parameter]], List[dict]]
) -> List[dict]:
    """
    为参数组设置 weight decay 策略。支持两种输入模式：

    1. 输入是 [(name, param)]：从头构建带 wd 的 param_groups
    2. 输入是 [{'params': [...], 'lr': ...}]：在已有分组上拆分 with/without decay

    Args:
        cfg: 配置对象，需包含 TRAIN.WITHOUT_WD_LIST, TRAIN.WD, VERBOSE 等
        model: 模型实例
        param_groups_or_named_params:
            - list of (name, param)，或
            - list of dict (如 optimizer 的 param_groups 结构)

    Returns:
        List[dict]: 带 weight_decay 字段的新参数组列表
    """
    if len(param_groups_or_named_params) == 0:
        return []

    # 判断输入类型
    is_param_groups = isinstance(param_groups_or_named_params[0], dict)

    # 构建 param -> name 映射，便于后续判断
    param_to_name = {}
    for name, param in model.named_parameters():
        param_to_name[param] = name

    # Step 1: 收集特殊层的参数（depthwise, norm）
    without_decay_depthwise = set()
    without_decay_norm = set()

    def _is_depthwise(m):
        return isinstance(m,
                          nn.Conv2d) and m.groups != 1 and m.groups == m.in_channels and m.out_channels % m.in_channels == 0

    for m in model.modules():
        # Depthwise Conv
        if _is_depthwise(m) and 'dw' in cfg.TRAIN.WITHOUT_WD_LIST:
            if m.weight is not None:
                without_decay_depthwise.add(m.weight)
        # BatchNorm
        elif isinstance(m, nn.BatchNorm2d) and 'bn' in cfg.TRAIN.WITHOUT_WD_LIST:
            if m.weight is not None:
                without_decay_norm.add(m.weight)
            if m.bias is not None:
                without_decay_norm.add(m.bias)
        # GroupNorm
        elif isinstance(m, nn.GroupNorm) and 'gn' in cfg.TRAIN.WITHOUT_WD_LIST:
            if m.weight is not None:
                without_decay_norm.add(m.weight)
            if m.bias is not None:
                without_decay_norm.add(m.bias)
        # LayerNorm
        elif isinstance(m, nn.LayerNorm) and 'ln' in cfg.TRAIN.WITHOUT_WD_LIST:
            if m.weight is not None:
                without_decay_norm.add(m.weight)
            if m.bias is not None:
                without_decay_norm.add(m.bias)

    # Step 2: 获取模型自定义规则
    skip = set()
    if hasattr(model, 'no_weight_decay') and callable(model.no_weight_decay):
        skip.update(model.no_weight_decay())

    skip_keywords = []
    if hasattr(model, 'no_weight_decay_keywords') and callable(model.no_weight_decay_keywords):
        skip_keywords = model.no_weight_decay_keywords()

    # Step 3: 处理输入并生成新 param_groups
    new_param_groups = []

    if is_param_groups:
        # 输入是 param_groups，例如 [{'params': [...], 'lr': 0.001}, ...]
        for group in param_groups_or_named_params:
            with_decay = []
            without_decay = []

            for param in group['params']:
                if not param.requires_grad:
                    continue

                name = param_to_name.get(param)
                if name is None:
                    # 安全兜底：如果找不到 name，默认加 WD
                    with_decay.append(param)
                    continue

                should_skip_wd = False

                # 规则1: 名称完全匹配或部分匹配 skip 集合
                if name in skip:
                    should_skip_wd = True
                else:
                    for key in skip:
                        if key in name:
                            should_skip_wd = True
                            break

                # 规则2: 匹配 no_weight_decay_keywords
                if not should_skip_wd and skip_keywords:
                    for kw in skip_keywords:
                        if kw in name:
                            should_skip_wd = True
                            break

                # 规则3: 是 depthwise 或 norm 层的参数
                if not should_skip_wd:
                    if param in without_decay_depthwise or param in without_decay_norm:
                        should_skip_wd = True

                # 规则4: bias 参数
                if not should_skip_wd:
                    if 'bias' in cfg.TRAIN.WITHOUT_WD_LIST and name.endswith('.bias'):
                        should_skip_wd = True

                # 分组
                if should_skip_wd:
                    without_decay.append(param)
                    if cfg.VERBOSE:
                        print(f'=> set {name} wd to 0')
                else:
                    with_decay.append(param)

            # 保留原 group 的超参（lr 等），只替换 params 并添加 wd
            base_group = {k: v for k, v in group.items() if k != 'params' and k != 'weight_decay'}

            if with_decay:
                new_group = base_group.copy()
                new_group['params'] = with_decay
                new_group['weight_decay'] = cfg.TRAIN.WD
                new_param_groups.append(new_group)

            if without_decay:
                new_group = base_group.copy()
                new_group['params'] = without_decay
                new_group['weight_decay'] = 0.0
                new_param_groups.append(new_group)

    else:
        # 输入是 [(name, param)] 列表，从头构建两个组
        with_decay = []
        without_decay = []

        for name, param in param_groups_or_named_params:
            if not param.requires_grad:
                continue

            should_skip_wd = False

            if name in skip or any(key in name for key in skip):
                should_skip_wd = True
            elif skip_keywords and any(kw in name for kw in skip_keywords):
                should_skip_wd = True
            elif param in without_decay_depthwise or param in without_decay_norm:
                should_skip_wd = True
            elif 'bias' in cfg.TRAIN.WITHOUT_WD_LIST and name.endswith('.bias'):
                should_skip_wd = True

            if should_skip_wd:
                without_decay.append(param)
                if cfg.VERBOSE:
                    print(f'=> set {name} wd to 0')
            else:
                with_decay.append(param)

        # 构建标准 param_groups
        if with_decay:
            new_param_groups.append({
                'params': with_decay,
                'weight_decay': cfg.TRAIN.WD
            })
        if without_decay:
            new_param_groups.append({
                'params': without_decay,
                'weight_decay': 0.0
            })

    return new_param_groups

def build_optimizer(cfg, model, params=None, is_train=True):
    """
    构建优化器

    Args:
        cfg: 配置
        model: 模型
        params: 可选，[(name, param)] 列表。如果为 None，则优化整个模型
        is_train: 是否训练模式（决定使用哪个学习率）
    """
    lr = cfg.TRAIN.LR
    if params is None:

        classifier_params = []
        backbone_params = []

        for name, param in model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': lr * 0.5},
            {'params': classifier_params, 'lr': lr}  # 分类器大学习率
        ]
        param_groups = set_wd(cfg, model, param_groups)
    else:
        # 设置 weight decay
        params = set_wd(cfg, model, params)
        param_groups = params

    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=True
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=cfg.TRAIN.WD
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamW':
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=cfg.TRAIN.WD,
            betas=(0.9, 0.98),
            eps=1e-6
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.TRAIN.OPTIMIZER}")

    return optimizer