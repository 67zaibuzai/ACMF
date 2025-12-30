import argparse
import logging
import os
import pprint
import time

import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.collect_env import get_pretty_env_info
from tensorboardX import SummaryWriter

from config import config
from config import update_config
from config import save_config
from core.loss import build_criterion
from core.function import dtod_finetune_one_epoch, ioe_test, ioe_train_one_epoch
from dataloders import create_dataloader
from models import build_model
from opim import build_optimizer
from scheduler import build_lr_scheduler
from utils.comm import comm
from utils.utils import create_logger, set_seed, save_queue_checkpoint_on_master, send_email
from utils.utils import init_distributed
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master
from utils.utils import resume_checkpoint
from utils.utils import save_checkpoint_on_master
from utils.utils import save_model_on_master

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

    # --cfg cfg path
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def setup_training_environment(args):
    """初始化训练环境"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    init_distributed(args)
    setup_cudnn(config)
    update_config(config, args)

    final_output_dir = create_logger(config, 'train')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))
        save_config(config, output_config_path)

    set_seed(config.SEED)

    return final_output_dir, tb_log_dir

def build_components(config, args, final_output_dir, is_finetune=False):
    """构建模型、优化器等训练组件"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建模型
    model = build_model(config, config.MODEL.NAME)
    model.to(device)
    summary_model_on_master(model, config, final_output_dir, True)

    if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nhwc':
        logging.info('=> convert memory format to nhwc')
        model.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(config, model)
    best_perf, begin_epoch = resume_checkpoint(
        model, optimizer, config, final_output_dir, True
    )
    lr_scheduler = build_lr_scheduler(config, optimizer, begin_epoch)
    scaler = torch.amp.GradScaler('cuda', enabled=config.AMP.ENABLED)

    # 数据加载器
    loaders = create_dataloader(config, is_train=True, distributed=args.distributed)
    train_dataloader = loaders['train']
    finetune_dataloader = loaders['finetune']
    test_dataloader = loaders['test']

    # 分布式训练
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    # 损失函数
    criterion = build_criterion(config, is_train=True).to(device)
    criterion_finetune = build_criterion(config, is_train=False).to(device)

    return {
        'model': model,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'scaler': scaler,
        'train_dataloader': train_dataloader,
        'finetune_dataloader': finetune_dataloader,
        'test_dataloader': test_dataloader,
        'criterion': criterion,
        'criterion_finetune': criterion_finetune,
        'device': device,
        'best_perf': best_perf,
        'begin_epoch': begin_epoch
    }


def initialize_queue(config, final_output_dir, device):
    """初始化或加载队列"""
    queue_fusion = None
    queue_path = os.path.join(final_output_dir, "final_queue.pth")

    if os.path.isfile(queue_path) and config.MODEL.SPEC.USE_NEIGHBOR_CLS:
        logging.info(f"=> loaded queue from {queue_path}")
        queue_fusion = torch.load(queue_path)["queue"]
        return queue_fusion

    config.defrost()
    config.QUEUE.QUEUE_LENGTH -= config.QUEUE.QUEUE_LENGTH % (
            config.TRAIN.BATCH_SIZE_PER_GPU * config.WORLD_SIZE
    )
    config.freeze()

    return queue_fusion


def create_queue_if_needed(config, epoch, queue_fusion, device):
    """在需要时创建队列"""
    if queue_fusion is not None or config.MODEL.SPEC.USE_NEIGHBOR_CLS:
        return queue_fusion
    if (config.QUEUE.QUEUE_LENGTH > 0 and
            epoch >= config.QUEUE.START_EPOCH and
            config.MODEL.SPEC.USE_NEIGHBOR_CLS):
        queue_fusion = torch.zeros(
            config.QUEUE.QUEUE_LENGTH // config.WORLD_SIZE,
            config.MODEL.SPEC.OUTPUT_DIM
        ).to(device)
        logging.info("=> Queue created")

    if queue_fusion is None:
        logging.info("=> Not using queue")

    return queue_fusion

def save_checkpoint(model, optimizer, epoch, best_perf, best_model,
                   queue_fusion, args, config, final_output_dir):
    """保存检查点"""
    save_checkpoint_on_master(
        model=model,
        distributed=args.distributed,
        model_name=config.MODEL.NAME,
        optimizer=optimizer,
        output_dir=final_output_dir,
        in_epoch=True,
        epoch_or_step=epoch,
        best_perf=best_perf,
        best=best_model
    )

    if best_model and comm.is_main_process():
        logging.info(f"=> saving best model checkpoint at epoch {epoch+1}")
        save_model_on_master(
            model, args.distributed, final_output_dir, 'model_best.pth'
        )


def unsupervised_training_stage(config, components, args, final_output_dir,
                                writer_dict, start_epoch, end_epoch):
    logging.info('=' * 80)
    logging.info('=> [STAGE 1: UNSUPERVISED TRAINING]')
    logging.info(f'=> Epochs: {start_epoch} to {end_epoch}')
    logging.info('=' * 80)

    model = components['model']
    optimizer = components['optimizer']
    lr_scheduler = components['lr_scheduler']
    scaler = components['scaler']
    train_dataloader = components['train_dataloader']
    criterion = components['criterion']
    device = components['device']

    # 加载或构建queue
    queue_fusion = initialize_queue(config, final_output_dir, device)
    best_perf = 0.0

    for epoch in range(start_epoch, end_epoch):
        head = f'[Unsupervised] Epoch[{epoch}/{end_epoch}]'
        logging.info(f'=> {head} start')
        start_time = time.time()

        # 设置随机种子
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        # 创建队列
        queue_fusion = create_queue_if_needed(config, epoch, queue_fusion, device)

        # 训练一个epoch
        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            queue_fusion = ioe_train_one_epoch(
                config, train_dataloader, model, criterion, optimizer,
                epoch, final_output_dir, final_output_dir, queue_fusion,
                writer_dict, scaler=scaler, device=device, end_epoch=end_epoch
            )

        # 更新学习率
        lr_scheduler.step(epoch=epoch + 1)
        if config.TRAIN.LR_SCHEDULER.METHOD == 'timm':
            lr = lr_scheduler._get_lr(epoch + 1)[0]
        else:
            lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        # 保存检查点
        save_checkpoint(
            model, optimizer, epoch, best_perf, False,
            queue_fusion, args, config, final_output_dir
        )

        duration = time.time() - start_time
        logging.info(f'=> {head} end, duration: {duration:.2f}s')

    # 保存无监督阶段的最终模型
    logging.info("=> Saving unsupervised stage model")
    save_model_on_master(
        model, args.distributed, final_output_dir, 'unsupervised_final.pth'
    )
    save_queue_checkpoint_on_master(
        queue_fusion, final_output_dir, name='unsupervised_queue.pth'
    )

    return queue_fusion


def supervised_training_stage(config, components, args, final_output_dir,
                              writer_dict, start_epoch, end_epoch, queue_fusion):
    """有监督训练阶段"""
    logging.info('=' * 80)
    logging.info('=> [STAGE 2: SUPERVISED FINE-TUNING]')
    logging.info(f'=> Epochs: {start_epoch} to {end_epoch}')
    logging.info('=' * 80)

    model = components['model']
    optimizer = components['optimizer']
    lr_scheduler = components['lr_scheduler']
    scaler = components['scaler']
    finetune_dataloader = components['finetune_dataloader']
    test_dataloader = components['test_dataloader']
    criterion_finetune = components['criterion_finetune']
    device = components['device']

    supervised_lr_decay = config.TRAIN.get('SUPERVISED_LR_DECAY', 0.1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * supervised_lr_decay
    logging.info(f'=> Finetuning with lr: {optimizer.param_groups[0]["lr"]}')

    best_perf = 0.0
    best_metrics = None
    no_improve_count = 0
    patience = config.TRAIN.get('EARLY_STOP_PATIENCE', 10)

    for epoch in range(start_epoch, end_epoch):
        head = f'[Supervised] Epoch[{epoch}/{end_epoch}]'
        logging.info(f'=> {head} start')
        start_time = time.time()

        # 设置随机种子
        if args.distributed:
            finetune_dataloader.sampler.set_epoch(epoch)

        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            perf, queue_fusion = dtod_finetune_one_epoch(
                config, finetune_dataloader, model, criterion_finetune,
                optimizer, epoch, final_output_dir, final_output_dir,
                writer_dict, scaler=scaler, device=device,
                queue=queue_fusion, end_epoch=end_epoch
            )

        # 评估性能
        best_model = False
        if perf > best_perf:
            best_perf = perf
            best_model = True
            no_improve_count = 0

            # 在测试集上评估
            metrics = ioe_test(
                test_dataloader, model, device=device, queue=queue_fusion
            )
            best_metrics = metrics
            logging.info(f'=> New best performance: {best_perf:.4f}')

            # 保存检查点
            save_checkpoint(
                model, optimizer, epoch, best_perf, best_model,
                queue_fusion, args, config, final_output_dir
            )
        else:
            no_improve_count += 1
            logging.info(f'=> No improvement for {no_improve_count} epochs')

        # 更新学习率
        lr_scheduler.step(epoch=epoch + 1)
        if config.TRAIN.LR_SCHEDULER.METHOD == 'timm':
            lr = lr_scheduler._get_lr(epoch + 1)[0]
        else:
            lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        # 保存检查点
        save_checkpoint(
            model, optimizer, epoch, best_perf, best_model,
            queue_fusion, args, config, final_output_dir
        )


        duration = time.time() - start_time
        logging.info(f'=> {head} end, duration: {duration:.2f}s')

        # Early stopping
        if no_improve_count >= patience:
            logging.info(f'=> Early stopping at epoch {epoch + 1}')
            break

    save_queue_checkpoint_on_master(
        queue_fusion, final_output_dir
    )
    return best_metrics
def main():
    args = parse_args()

    queue_fusion = None

    # 1. 设置训练环境
    final_output_dir, tb_log_dir = setup_training_environment(args)

    train_end_epoch = config.TRAIN.END_EPOCH
    test_end_epoch = config.TEST.END_EPOCH

    # 2. 构建训练组件
    supervised_output_dir = os.path.join(final_output_dir, "supervised")
    if os.path.exists(supervised_output_dir):
        components = build_components(config, args, supervised_output_dir)

    else:
        components = build_components(config, args, final_output_dir)

        writer_dict = {
            'writer': SummaryWriter(logdir=tb_log_dir),
            'train_global_steps': components['begin_epoch'],
            'valid_global_steps': components['begin_epoch'],
        }

        # 无监督训练
        logging.info(f"=============无监督训练==============")
        queue_fusion = unsupervised_training_stage(config=config,
                                                   components=components,
                                                   args=args,
                                                   final_output_dir=final_output_dir,
                                                   writer_dict=writer_dict,
                                                   start_epoch=components['begin_epoch'],
                                                   end_epoch=train_end_epoch,)
    logging.info(f"===============有监督训练=============")

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': components['begin_epoch'],
        'valid_global_steps': components['begin_epoch'],
    }

    model = components['model']
    optimizer = build_optimizer(config, model)

    if os.path.exists(supervised_output_dir):
        best_perf, start_epoch = resume_checkpoint(
            model, optimizer, config, supervised_output_dir, True
        )
    else:
        os.makedirs(supervised_output_dir)
        start_epoch = 0

    scheduler = build_lr_scheduler(config, optimizer, begin_epoch=0)
    components['optimizer'] = optimizer
    components['lr_scheduler'] = scheduler
    components['scaler'] = torch.amp.GradScaler('cuda', enabled=config.AMP.ENABLED)

    metrics = supervised_training_stage(
        config=config,
        components=components,
        args=args,
        final_output_dir=supervised_output_dir,
        writer_dict=writer_dict,
        start_epoch=start_epoch,
        end_epoch=test_end_epoch,
        queue_fusion=queue_fusion,
    )

    # 保存最终模型
    logging.info("=> Saving final model checkpoint")
    save_model_on_master(
        components['model'], args.distributed, final_output_dir, 'final_state.pth'
    )
    save_queue_checkpoint_on_master(
        queue_fusion, final_output_dir, name='final_queue.pth'
    )

    csv_path = os.path.join(final_output_dir, 'metrics.csv')

    if metrics:
        new_row = pd.DataFrame([metrics])

        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            df = new_row

        df.to_csv(csv_path, index=False)
        logging.info(f"=> Metrics saved to {csv_path}")

    writer_dict['writer'].close()
    logging.info('=> Training finished!')

    send_email(
        subject="【训练完成】你的模型已训练完毕！",
        body=f"你好！\n\n你的模型训练已经顺利完成。\n{metrics}\n加油加油加加油",
    )
if __name__ == '__main__':
    main()
