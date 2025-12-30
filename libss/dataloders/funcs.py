import torch

def split_val_sets(val_class_0, val_class_1, fine_tune_ratio=0.8):
    """
    在val_class_0和val_class_1中按比例分割
    fine_tune_ratio: 微调验证集的比例（默认80%）
    """
    fine_tune_val_0_size = int(len(val_class_0) * fine_tune_ratio)  # 199 * 0.8 ≈ 159
    fine_tune_val_1_size = int(len(val_class_1) * fine_tune_ratio)  # 51 * 0.8 ≈ 41

    fine_tune_val_0 = val_class_0[:fine_tune_val_0_size]  # 前80%作为微调验证集
    test_val_0 = val_class_0[fine_tune_val_0_size:]  # 后20%作为测试验证集

    fine_tune_val_1 = val_class_1[:fine_tune_val_1_size]  # 前80%作为微调验证集
    test_val_1 = val_class_1[fine_tune_val_1_size:]  # 后20%作为测试验证集

    # 合并得到最终的索引列表
    val_finetune_indices = fine_tune_val_0 + fine_tune_val_1
    val_test_indices = test_val_0 + test_val_1

    return val_finetune_indices, val_test_indices



