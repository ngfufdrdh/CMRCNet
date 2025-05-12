import torch
import numpy as np
import random
import logging
import os
from datetime import datetime

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_logger(cancer):
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_filename = f'{current_time}_train.log'
    log_dir = os.path.join('./logs', cancer)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            # logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()
    return logger


def print_and_log_output(logger, optimizer, args, effective_batch_size, device):

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Initial Learning rate: {args.lr}")
    logger.info(f"Max learning rate: {args.max_lr}")
    logger.info(f"Min learning rate: {args.min_lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Accumulation steps: {args.accumulation_steps}")
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Device: {device}")

    print(f"Optimizer: {optimizer}")
    print(f"Initial Learning rate: {args.lr}")
    print(f"Max learning rate: {args.max_lr}")
    print(f"Min learning rate: {args.min_lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Accumulation steps: {args.accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Device: {device}")

