from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import torch
from timm.data import create_transform
import logging
import torchvision.transforms as T
from torch import nn
from torchvision.transforms.v2.functional import to_tensor

def build_image_transforms(cfg, is_train):
    if cfg.AUG.TIMM_AUG.USE_TRANSFORM and is_train:
        logging.info('=> use timm transform for training')
        timm_cfg = cfg.AUG.TIMM_AUG
        transforms = create_transform(
            input_size=cfg.TRAIN.IMAGE_SIZE[0],
            is_training=True,
            use_prefetcher=False,
            no_aug=False,
            re_prob=timm_cfg.RE_PROB,
            re_mode=timm_cfg.RE_MODE,
            re_count=timm_cfg.RE_COUNT,
            scale=cfg.AUG.SCALE,
            ratio=cfg.AUG.RATIO,
            hflip=timm_cfg.HFLIP,
            vflip=timm_cfg.VFLIP,
            auto_augment=timm_cfg.AUTO_AUGMENT,
            interpolation=timm_cfg.INTERPOLATION,
            mean=cfg.AUG.MEAN,
            std=cfg.AUG.STD,
        )

        return transforms

    normalize = T.Normalize(mean=cfg.AUG.MEAN, std=cfg.AUG.STD)

    transforms = None
    if is_train:
        if cfg.FINETUNE.FINETUNE and not cfg.FINETUNE.USE_TRAIN_AUG:
            # precrop, crop = get_resolution(cfg.TRAIN.IMAGE_SIZE)
            crop = cfg.TRAIN.IMAGE_SIZE[0]
            precrop = crop + 32
            transforms = T.Compose([
                T.Resize(
                    (precrop, precrop),
                    interpolation=cfg.AUG.INTERPOLATION
                ),
                T.RandomCrop((crop, crop)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
        else:
            aug = cfg.AUG
            scale = aug.SCALE
            ratio = aug.RATIO
            ts = [T.RandomResizedCrop(
                cfg.TRAIN.IMAGE_SIZE[0], scale=scale, ratio=ratio,
                interpolation=cfg.AUG.INTERPOLATION
            ), T.RandomHorizontalFlip(), T.ToTensor(), normalize]

            transforms = T.Compose(ts)
    else:
        if cfg.TEST.CENTER_CROP:
            transforms = T.Compose([
                T.Resize(
                    int(cfg.TEST.IMAGE_SIZE[0] / 0.875),
                    interpolation=cfg.TEST.INTERPOLATION
                ),
                T.CenterCrop(cfg.TEST.IMAGE_SIZE[0]),
                T.ToTensor(),
                normalize,
            ])
        else:
            transforms = T.Compose([
                T.Resize(
                    (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0]),
                    interpolation=cfg.TEST.INTERPOLATION
                ),
                T.ToTensor(),
                normalize,
            ])

    return transforms



def build_aug_image_transformer(cfg):
    timm_cfg = cfg.AUG.TIMM_AUG
    transform_v1 = create_transform(
        input_size=cfg.TRAIN.IMAGE_SIZE[0],
        is_training=True,
        use_prefetcher=False,
        no_aug=False,
        re_prob=timm_cfg.RE_PROB,
        re_mode=timm_cfg.RE_MODE,
        re_count=timm_cfg.RE_COUNT,
        scale=(1.0, 1.0),
        ratio=cfg.AUG.RATIO,
        hflip=timm_cfg.HFLIP,
        vflip=timm_cfg.VFLIP,
        auto_augment=None,
        interpolation=timm_cfg.INTERPOLATION,
        mean=cfg.AUG.MEAN,
        std=cfg.AUG.STD,
    )

    transform_v2 = create_transform(
        input_size=cfg.TRAIN.IMAGE_SIZE[0],
        is_training=True,
        use_prefetcher=False,
        no_aug=False,
        re_prob=timm_cfg.RE_PROB,
        re_mode=timm_cfg.RE_MODE,
        re_count=timm_cfg.RE_COUNT,
        scale=cfg.AUG.SCALE,
        ratio=cfg.AUG.RATIO,
        hflip=timm_cfg.HFLIP,
        vflip=timm_cfg.VFLIP,
        auto_augment=timm_cfg.AUTO_AUGMENT,   # 开 AutoAugment
        interpolation=timm_cfg.INTERPOLATION,
        mean=cfg.AUG.MEAN,
        std=cfg.AUG.STD,
    )

    return {"v1": transform_v1, "v2": transform_v2,}


class TableNormalize(nn.Module):
    """标准化表格数据 (z-score)"""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-6)


class TableMinMaxScale(nn.Module):
    """Min-Max 归一化到 [0, 1]"""
    def __init__(self, min_val, max_val):
        super().__init__()
        self.register_buffer("min_val", torch.tensor(min_val, dtype=torch.float32))
        self.register_buffer("max_val", torch.tensor(max_val, dtype=torch.float32))

    def forward(self, x):
        return (x - self.min_val) / (self.max_val - self.min_val + 1e-6)


class TableRandomNoise(nn.Module):
    """训练时加随机噪声 (数据增强)"""
    def __init__(self, sigma=0.01):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        noise = torch.randn_like(x) * self.sigma
        return x + noise


def build_table_transforms(cfg, is_train):
    """
    构建表格数据的 transforms
    cfg: 配置对象
    is_train: 是否训练模式
    """
    transforms = []

    # 归一化方式
    if cfg.AUG.TABLE_NORM_TYPE == "zscore":
        transforms.append(TableNormalize(cfg.AUG.TABLE_MEAN, cfg.AUG.TABLE_STD))
    elif cfg.AUG.TABLE_NORM_TYPE == "minmax":
        transforms.append(TableMinMaxScale(cfg.AUG.TABLE_MEAN, cfg.AUG.TABLE_STD))

    # 训练增强
    if is_train and cfg.AUG.USE_NOISE:
        transforms.append(TableRandomNoise(cfg.AUG.NOISE_STD))

    return nn.Sequential(*transforms)

class TableCompose(nn.Module):
    """像 torchvision.transforms.Compose 一样串联表格增强"""
    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x):
        for t in self.transforms:
            x = t(x)
        return x

def build_aug_table_transformer(cfg, is_train=True):
    mean = cfg.AUG.TABLE_MEAN
    std = cfg.AUG.TABLE_STD

    transform_v1 = TableCompose([
        TableNormalize(mean, std),
    ])

    # View2: 轻微扰动
    transform_v2 = TableCompose([
        TableNormalize(mean, std),
        TableRandomNoise(sigma=cfg.AUG.NOISE_SIGMA1),
    ])

    return {"v1": transform_v1, "v2": transform_v2}


def build_transforms(cfg, is_train=True, type='imagetable'):
    if type == 'image':
        image_transform = build_image_transforms(cfg, is_train)
        return image_transform
    elif type == 'option':
        table_transform = build_table_transforms(cfg, is_train)
        return table_transform

    elif type == 'semantic':
        table_transform = build_table_transforms(cfg, is_train)
        return table_transform

    elif type == 'table':
        table_transform = build_table_transforms(cfg, is_train)
        return table_transform
    else:
        image_transforms = build_aug_image_transformer(cfg)
        table_transforms = build_aug_table_transformer(cfg)
        return image_transforms, table_transforms
