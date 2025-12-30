from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from datetime import timedelta
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.header import Header

import os
import logging
import shutil
import time

import numpy as np
from torchinfo import summary
from thop import profile, clever_format

import torch
import torch.backends.cudnn as cudnn

from .comm import comm
from ptflops import get_model_complexity_info


def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_rank{}.txt'.format(phase, time_str, rank)
    # 构建文件名
    final_log_file = os.path.join(final_output_dir, log_file)
    # 构建logger格式
    head = '%(asctime)-15s:[P:%(process)d]:' + comm.head + ' %(message)s'
    logging.basicConfig(
        filename=str(final_log_file), format=head
    )
    logger = logging.getLogger()
    # 日志级别设为 INFO，即只记录 INFO、WARNING、ERROR、CRITICAL，忽略 DEBUG。
    logger.setLevel(logging.INFO)
    # 新增一个控制台输出
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(head)
    )
    logging.getLogger('').addHandler(console)


def create_logger(cfg, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    model = cfg.MODEL.NAME
    dataset = cfg.DATASET.DATASET
    cfg_name = cfg.NAME

    final_output_dir = root_output_dir / model / dataset / cfg_name

    print('=> creating {} ...'.format(root_output_dir))
    root_output_dir.mkdir(parents=True, exist_ok=True)
    print('=> creating {} ...'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    print('=> setup logger ...')
    setup_logger(final_output_dir, cfg.RANK, phase)

    return str(final_output_dir)


def init_distributed(args):
    args.num_gpus = torch.cuda.device_count()
    args.distributed = args.num_gpus > 1

    print(f"GPU count: {args.num_gpus}")
    if args.distributed:
        print("=> init process group start")
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(minutes=180))
        comm.local_rank = args.local_rank
        print("=> init process group end")


def setup_cudnn(config):
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def summary_model_on_master(model, config, output_dir, name=None):
    if name is not None:
        name = config.MODEL.NAME
    if comm.is_main_process():
        this_dir = os.path.dirname(__file__)
        # 复制一份模型文件
        shutil.copy2(
            os.path.join(this_dir, '../models', name + '.py'),
            output_dir
        )
        logging.info('=> {}'.format(model))
        try:
            num_params = count_parameters(model)
            logging.info("Trainable Model Total Parameter: \t%2.1fM" % num_params)
        except Exception:
            logging.error('=> error when counting parameters')

        if config.MODEL_SUMMARY:
            try:
                # 保存模型的原始设备
                original_device = next(model.parameters()).device

                # 暂时将模型移到 CPU 进行分析
                model_cpu = model.cpu()
                input_shape = (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
                dummy_input = torch.randn(input_shape)  # 在 CPU 上创建输入

                # 使用 torchinfo
                model_summary = summary(
                    model_cpu,  # 使用 CPU 上的模型
                    input_size=input_shape,
                    verbose=0,
                    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"]
                )

                logging.info('== Model Summary ==')
                summary_path = os.path.join(output_dir, "model_summary.txt")  # 可以自定义路径和文件名
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write("== Model Summary ==\n")
                    f.write(str(model_summary))

                # 将模型恢复回原来的设备
                model = model.to(original_device)

            except ImportError:
                logging.warning('torchinfo or thop not installed, skipping model summary')
            except Exception as e:
                logging.error(f'Model summary error: {e}')

            try:
                logging.info('== get_model_complexity_info by ptflops ==')
                macs, params = get_model_complexity_info(
                    model,
                    (3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]),
                    as_strings=True, print_per_layer_stat=True, verbose=True
                )
                logging.info(f'=> FLOPs: {macs:<8}, params: {params:<8}')
                logging.info('== get_model_complexity_info by ptflops ==')
            except Exception:
                logging.error('=> error when run get_model_complexity_info')

def set_seed(seed=42):
    """设置所有随机种子以确保可重复性"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resume_checkpoint(model,
                      optimizer,
                      config,
                      output_dir,
                      in_epoch,
                      model_name=None):
    best_perf = -10000.0
    begin_epoch_or_step = 0
    if model_name is None:
        model_name = config.MODEL.NAME
    checkpoint = os.path.join(output_dir, f'{model_name}_checkpoint.pth')\
        if not config.TRAIN.CHECKPOINT else config.TRAIN.CHECKPOINT
    if config.TRAIN.AUTO_RESUME and os.path.exists(checkpoint):
        logging.info(
            "=> loading checkpoint '{}'".format(checkpoint)
        )
        checkpoint_dict = torch.load(checkpoint, map_location='cpu')
        best_perf = checkpoint_dict.get('perf')
        begin_epoch_or_step = checkpoint_dict['epoch' if in_epoch else 'step']

        model.load_state_dict(checkpoint_dict['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
        logging.info(
            "=> {}: loaded checkpoint '{}' ({}: {})"
            .format(comm.head,
                    checkpoint,
                    'epoch' if in_epoch else 'step',
                    begin_epoch_or_step)
        )

    return best_perf, begin_epoch_or_step


def save_checkpoint_on_master(model,
                              distributed,
                              model_name,
                              optimizer,
                              output_dir,
                              in_epoch,
                              epoch_or_step,
                              best_perf,
                              best = False,
                              finetune_optimizer=None):
    if not comm.is_main_process():
        return

    states = model.module.state_dict() \
        if distributed else model.state_dict()

    logging.info('=> saving checkpoint to {}'.format(output_dir))
    save_dict = {
        'epoch' if in_epoch else 'step': epoch_or_step + 1,
        'model': model_name,
        'state_dict': states,
        'perf': best_perf,
        'optimizer': optimizer.state_dict(),
        'finetune_optimizer': finetune_optimizer.state_dict() if finetune_optimizer is not None else None
    }

    try:
        if not best:
            torch.save(save_dict, os.path.join(output_dir, f'{model_name}_checkpoint.pth'))
        else:
            torch.save(save_dict, os.path.join(output_dir, f'{model_name}_checkpoint_best.pth'))
            torch.save(save_dict, os.path.join(output_dir, f'{model_name}_checkpoint.pth'))
    except Exception:
        logging.error('=> error when saving checkpoint!')

def save_queue_checkpoint_on_master(queue, output_dir, name='queue.pth'):
    if not comm.is_main_process():
        return

    logging.info('=> saving queue to {}'.format(output_dir))
    save_dict = {
        'queue':queue,
    }
    try:
        torch.save(save_dict, os.path.join(output_dir, name))
    except Exception:
        logging.error('=> error when saving checkpoint!')

def save_model_on_master(model, distributed, out_dir, fname):
    if not comm.is_main_process():
        return

    try:
        fname_full = os.path.join(out_dir, fname)
        logging.info(f'=> save model to {fname_full}')
        torch.save(
            model.module.state_dict() if distributed else model.state_dict(),
            fname_full
        )
    except Exception:
        logging.error('=> error when saving checkpoint!')


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    from collections import OrderedDict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def send_email(subject, body, to_addr="2146981074@qq.com", from_addr="2146981074@qq.com", password="npickdxvuabbbedd", smtp_server='smtp.qq.com', port=587):
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['From'] = Header(from_addr)
    msg['To'] = Header(to_addr)
    msg['Subject'] = Header(subject)

    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()  # 启用 TLS 加密
        server.login(from_addr, password)
        server.sendmail(from_addr, [to_addr], msg.as_string())
        server.quit()
        print("📧 邮件发送成功！")
    except Exception as e:
        print(f"❌ 邮件发送失败：{e}")