import logging
import math

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset, Sampler, SubsetRandomSampler, SequentialSampler

import os
from PIL import Image
import pandas as pd
import numpy as np
import random

from .transforms import build_transforms
import torchvision.transforms as T
from .funcs import split_val_sets

#todo: 多模态数据的加载
#todo: 下一篇融入多样性特征、聚集性特征和对称性特征等

class ImagetableDataset(Dataset):
    def __init__(self, cfg, transform=None):
        self.img_dir = cfg.IMAGE_PATH
        self.option_path = getattr(cfg, 'OPTION_PATH', None)
        self.semantic_path = getattr(cfg, 'SEMANTIC_PATH', None)
        self.label_path = cfg.LABEL_PATH
        self.transform = transform

        # 检查必要路径
        if not os.path.exists(self.label_path):
            raise FileNotFoundError('Label path does not exist')
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError('Image path does not exist')
        if not os.path.exists(self.semantic_path):
            raise FileNotFoundError('Semantic path does not exist')
        if not os.path.exists(self.option_path):
            raise FileNotFoundError('Operation path does not exist')

        self.use_image =cfg.MODEL.SPEC.USE_IMAGE
        self.use_option = cfg.MODEL.SPEC.USE_OPE
        self.use_semantic = cfg.MODEL.SPEC.USE_EMO

        # 加载标签
        label_df = self.tackle_labels(self.label_path)
        self.label_dict = dict(zip(label_df.iloc[:, 0].astype(str), label_df.iloc[:, 1]))

        # 加载 option 特征（如果启用）
        option_dict = {}
        if self.use_option:
            option_df = pd.read_excel(self.option_path, header=None)
            for _, row in option_df.iterrows():
                key_str = str(row[0])
                if key_str.replace('.', '').isdigit():
                    key = str(int(float(key_str)))
                    option_dict[key] = row[1:].values.astype(np.float32)

        # 加载 semantic 特征（如果启用）
        semantic_dict = {}
        if self.use_semantic:
            semantic_df = pd.read_excel(self.semantic_path, header=None)
            for _, row in semantic_df.iterrows():
                key_str = str(row[0])
                if key_str.isdigit():
                    semantic_dict[key_str] = row[1:].values.astype(np.float32)

        self.samples = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

        for fname in os.listdir(self.img_dir):
            if not fname.lower().endswith(valid_extensions):
                continue
            if fname.startswith('._') or fname == '.DS_Store':
                continue

            name = os.path.splitext(fname)[0]  # 安全地去掉扩展名
            img_path = os.path.join(self.img_dir, fname)

            # 验证图像是否可读（提前过滤）
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                continue

            label = self.label_dict.get(name, -2)
            if label == -2:
                continue

            option_feat = option_dict.get(name, None) if self.use_option else None
            semantic_feat = semantic_dict.get(name, None) if self.use_semantic else None

            self.samples.append((img_path, option_feat, semantic_feat, label))

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found!")

        self.labels = np.array([s[3] for s in self.samples], dtype=np.int64)

    def tackle_labels(self, label_path):
        if label_path.endswith('.csv'):
            labels_df = pd.read_csv(label_path)
        elif label_path.endswith(('.xlsx', '.xls')):
            labels_df = pd.read_excel(label_path)
        else:
            raise ValueError(f"Unsupported label file format: {label_path}")

        # 假设列名为 'sp_id' 和 'label'；若没有，用位置索引
        if 'sp_id' in labels_df.columns and 'label' in labels_df.columns:
            return labels_df[['sp_id', 'label']].copy()
        else:
            # 默认第0列为ID，第1列为label
            labels_df = labels_df.iloc[:, :2].copy()
            labels_df.columns = ['sp_id', 'label']
            return labels_df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, option_table, semantic_table, label = self.samples[idx]

        # 收集原始模态数据（只收集启用的）
        raw_views = []

        if self.use_image:
            img = Image.open(img_path).convert('RGB')
            raw_views.append(img)
        if self.use_option and option_table is not None:
            option_tensor = torch.tensor(option_table, dtype=torch.float32)
            raw_views.append(option_tensor)
        if self.use_semantic and semantic_table is not None:
            semantic_tensor = torch.tensor(semantic_table, dtype=torch.float32)
            raw_views.append(semantic_tensor)

        if len(raw_views) == 0:
            raise RuntimeError("No modality is enabled or available for this sample.")

        if self.transform is None:
            return tuple(raw_views) + (label,)

        img_tfs = self.transform[0]
        table_tf = self.transform[1] # 假设表格用同一个增强函数（可改为 v1/v2 分别处理）

        view1 = []
        view2 = []

        view_idx = 0
        if self.use_image:
            img = raw_views[view_idx]
            view1.append(img_tfs['v1'](img))
            view2.append(img_tfs['v2'](img))
            view_idx += 1

        if self.use_option and option_table is not None:
            opt = raw_views[view_idx]
            view1.append(table_tf['v1'](opt))
            view2.append(table_tf['v2'](opt))
            view_idx += 1

        if self.use_semantic and semantic_table is not None:
            sem = raw_views[view_idx]
            view1.append(table_tf['v1'](sem))
            view2.append(table_tf['v2'](sem))
            # view_idx += 1  # 不再需要

        return (
            tuple(view1),
            tuple(view2),
            label
        )
def guaranteed_triple_balanced_sampler(labeled_class_0, labeled_class_1, unlabeled_indices, batch_size,
                                       target_labeled_ratio=0.1):
    class StrictBalancedBatchSampler(Sampler):
        def __init__(self, labeled_class_0, labeled_class_1, unlabeled, batch_size, labeled_ratio=0.1):
            super().__init__()
            self.labeled_0 = labeled_class_0
            self.labeled_1 = labeled_class_1
            self.unlabeled = unlabeled
            self.batch_size = batch_size
            self.labeled_ratio = labeled_ratio

            # === 每个 batch 内的分配 ===
            self.labeled_per_batch = max(1, int(batch_size * labeled_ratio))
            self.unlabeled_per_batch = batch_size - self.labeled_per_batch

            # 类内比例：用实际数量比决定
            total_labeled = len(labeled_class_0) + len(labeled_class_1)
            self.class0_ratio = len(labeled_class_0) / total_labeled if total_labeled > 0 else 0.5
            self.class_0_per_batch = max(0, int(self.labeled_per_batch * self.class0_ratio))
            self.class_1_per_batch = self.labeled_per_batch - self.class_0_per_batch

            # === epoch 内可用的数据 ===
            self.labeled_0_epoch = self.labeled_0.copy()
            self.labeled_1_epoch = self.labeled_1.copy()
            self.unlabeled_epoch = self.unlabeled.copy()

            random.shuffle(self.labeled_0_epoch)
            random.shuffle(self.labeled_1_epoch)
            random.shuffle(self.unlabeled_epoch)

            # === 计算 epoch 大小 ===
            self.num_batches = min(
                len(self.labeled_0_epoch) // self.class_0_per_batch if self.class_0_per_batch > 0 else float('inf'),
                len(self.labeled_1_epoch) // self.class_1_per_batch if self.class_1_per_batch > 0 else float('inf'),
                len(self.unlabeled_epoch) // self.unlabeled_per_batch if self.unlabeled_per_batch > 0 else float('inf')
            )

        def __iter__(self):
            labeled_0_iter = iter(self.labeled_0_epoch)
            labeled_1_iter = iter(self.labeled_1_epoch)
            unlabeled_iter = iter(self.unlabeled_epoch)

            for _ in range(self.num_batches):
                batch = []
                for _ in range(self.class_0_per_batch):
                    batch.append(next(labeled_0_iter))
                for _ in range(self.class_1_per_batch):
                    batch.append(next(labeled_1_iter))
                for _ in range(self.unlabeled_per_batch):
                    batch.append(next(unlabeled_iter))
                yield batch

        def __len__(self):
            return self.num_batches

    return StrictBalancedBatchSampler(labeled_class_0, labeled_class_1, unlabeled_indices, batch_size, target_labeled_ratio)

def worker_init_fn(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def create_labeled_dataloader(cfg, is_train: bool, distributed=False, train_ratio=0.8):
    image_transforms, table_transforms = build_transforms(cfg)

    if is_train:
        batch_size_per_gpu = cfg.TRAIN.BATCH_SIZE_PER_GPU
        shuffle = True
    else:
        batch_size_per_gpu = cfg.TEST.BATCH_SIZE_PER_GPU
        shuffle = False

    dataset = ImagetableDataset(cfg, [image_transforms, table_transforms])

    # === 划分有标签 / 无标签 ===
    labeled_indices = [i for i, (_, _, label) in enumerate(dataset.samples) if label != -1]
    unlabeled_indices = [i for i, (_, _, label) in enumerate(dataset.samples) if label == -1]

    # --- 进一步划分有标签数据的类别 ---
    labeled_class_0 = [i for i in labeled_indices if dataset.samples[i][2] == 0]
    labeled_class_1 = [i for i in labeled_indices if dataset.samples[i][2] == 1]

    print(f"有标签数据 - 类别0: {len(labeled_class_0)}, 类别1: {len(labeled_class_1)}")
    print(f"无标签数据: {len(unlabeled_indices)}")

    val_class_0 = torch.randperm(len(labeled_class_0))[:199].tolist()
    val_class_1 = torch.randperm(len(labeled_class_1))[:51].tolist()

    val_indices = [labeled_class_0[i] for i in val_class_0] + [labeled_class_1[i] for i in val_class_1]

    val_class_0_global = [labeled_class_0[i] for i in val_class_0]
    val_class_1_global = [labeled_class_1[i] for i in val_class_1]
    val_finetune_indices, val_test_indices = split_val_sets(val_class_0_global, val_class_1_global, fine_tune_ratio=0.8)

    train_labeled_class_0 = [i for i in labeled_class_0 if i not in val_indices]
    train_labeled_class_1 = [i for i in labeled_class_1 if i not in val_indices]
    train_labeled_indices = train_labeled_class_0 + train_labeled_class_1

    # --- Dataset Subset ---
    # 创建包含所有训练数据（有标签+无标签）的数据集
    all_train_indices = train_labeled_indices + unlabeled_indices
    train_dataset = Subset(dataset, all_train_indices)
    val_finetune_dataset = Subset(dataset, val_finetune_indices)
    val_test_dataset = Subset(dataset, val_test_indices)

    # --- 构造全局→局部索引映射 ---
    idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(all_train_indices)}

    train_labeled_class_0 = [idx_map[i] for i in train_labeled_class_0]
    train_labeled_class_1 = [idx_map[i] for i in train_labeled_class_1]
    unlabeled_indices = [idx_map[i] for i in unlabeled_indices]

    if is_train:
        try:
            # 使用保证比例的采样器
            batch_sampler = guaranteed_triple_balanced_sampler(
                train_labeled_class_0,
                train_labeled_class_1,
                unlabeled_indices,
                batch_size=batch_size_per_gpu,
                target_labeled_ratio=len(train_labeled_indices) / len(all_train_indices)
            )

            sampler = None
            shuffle = False
            use_batch_sampler = True

        except ValueError as e:
            print(f"警告: {e}，使用默认采样器")
            batch_sampler = None
            sampler = None
            shuffle = True
            use_batch_sampler = False
    else:
        batch_sampler = None
        sampler = None
        use_batch_sampler = False

    # 分布式训练处理
    if distributed and sampler is not None:
        # 如果使用自定义采样器且需要分布式训练
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=False  # 自定义采样器已经处理了采样逻辑
        )
    elif distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle
        )

    # --- Dataloaders ---
    if is_train:
        if use_batch_sampler:
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=cfg.WORKERS,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size_per_gpu,
                shuffle=shuffle,
                num_workers=cfg.WORKERS,
                pin_memory=True,
                drop_last=True,
                sampler=sampler,
                worker_init_fn=worker_init_fn,
            )
        return train_loader
    else:
        val_finetune_loader = DataLoader(
            val_finetune_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=cfg.WORKERS,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )
        val_test_loader = DataLoader(
            val_test_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )
        return val_finetune_loader, val_test_loader

def create_dataloader(cfg, is_train: bool, distributed=False, finetune_ratio=0.7):
    image_transforms, table_transforms = build_transforms(cfg)
    dataset = ImagetableDataset(cfg, [image_transforms, table_transforms])

    labels = []
    labeled_indices = []
    unlabeled_indices = []

    for i, (_, _, _, label) in enumerate(dataset.samples):
        labels.append(label)
        if label != -1:
            labeled_indices.append(i)
        else:
            unlabeled_indices.append(i)

    def get_loader(indices, shuffle):
        subset = Subset(dataset, indices)
        local_indices = list(range(len(subset)))  # 0 ~ len(subset)-1
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(subset, shuffle=shuffle)
        else:
            sampler = torch.utils.data.SubsetRandomSampler(
                local_indices) if shuffle else torch.utils.data.SequentialSampler(subset)
        loader = DataLoader(
            subset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU if is_train else cfg.TEST.BATCH_SIZE_PER_GPU,
            sampler=sampler,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=(shuffle and len(indices) >= cfg.TRAIN.BATCH_SIZE_PER_GPU),
            worker_init_fn=worker_init_fn,
        )
        return loader

    dataloaders = {}

    # todo: 平衡训练集、测试集
    if is_train:
        np.random.shuffle(labeled_indices)
        all_labels = np.array(labels)  # 确保是 numpy 数组
        labeled_labels = all_labels[labeled_indices]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - finetune_ratio, train_size=finetune_ratio,
                                     random_state=42)

        for train_idx, test_idx in sss.split(np.zeros(len(labeled_indices)), labeled_labels):
            finetune_indices = [labeled_indices[i] for i in train_idx]
            test_indices = [labeled_indices[i] for i in test_idx]
        """split_idx = max(1, int(finetune_ratio * len(labeled_indices)))
        finetune_indices = labeled_indices[:split_idx]
        test_indices = labeled_indices[split_idx:]"""

        dataloaders['train'] = get_loader(unlabeled_indices + finetune_indices, shuffle=True)
        dataloaders['finetune'] = get_loader(finetune_indices, shuffle=True)
        dataloaders['test'] = get_loader(test_indices, shuffle=False)
    else:
        key = 'finetune' if getattr(cfg, 'EVAL_FINETUNE', False) else 'test'
        dataloaders[key] = get_loader(labeled_indices, shuffle=False)

    logging.info(f"Train size: {len(unlabeled_indices)} | Finetune size: {len(labeled_indices)}")

    return dataloaders

def create_graph_dataloader(cfg, batch_size, finetune_ratio=0.8, seed=42):
    dataset = ImagetableDataset(cfg, transform=build_transforms(cfg))

    all_indices = np.arange(len(dataset))  # 全部样本索引
    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)  # 前期打乱

    # 找到全部数据中有标签的样本（0/1）
    labeled_mask = (dataset.labels == 0) | (dataset.labels == 1)
    labeled_indices = np.where(labeled_mask)[0]  # 全部数据中有标签的索引
    labeled_labels = dataset.labels[labeled_indices]

    # 划分训练标签和测试标签
    split_idx = max(1, int(finetune_ratio * len(labeled_indices)))
    train_label_indices = labeled_indices[:split_idx]
    train_label_values = labeled_labels[:split_idx]
    test_label_indices = labeled_indices[split_idx:]
    test_label_values = labeled_labels[split_idx:]

    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=SequentialSampler(all_indices),  # 不再打乱
        shuffle=False, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY
    )


    return dataloader, (train_label_indices, train_label_values), (test_label_indices, test_label_values)



