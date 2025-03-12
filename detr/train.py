import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple
import logging

# Huggingface
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object
from safetensors.torch import load_model


@dataclass
class TrainingConfig:
    output_dir: str
    overwrite_output_dir: bool = field(default=True)  # overwrite the old model
    eval_only: bool = field(default=False)
    start_epoch: int = field(default=0)
    resume_from_checkpoint: str = field(default=None)

    coco_dataset_root: str = field(default="/media/bryan/ssd01/fiftyone/coco-2017")

    train_batch_size: int = field(default=128)
    val_batch_size: int = field(default=256)

    epochs: int = field(default=100)
    limit_train_iters: int = field(default=0)
    limit_val_iters: int = field(default=0)

    # Optimizer configuration
    optimizer_name: str = field(default="adamw")
    momentum: float = field(default=0.937)
    # Linear warmup + CosineAnnealingLR
    # 2e-4 for AdamW
    # 0.01-0.05 for SGD
    lr: float = field(default=2e-4)
    lr_warmup_epochs: int = field(default=5)
    lr_warmup_decay: float = field(default=0.01)
    lr_min: float = field(default=0.0)

    # Regularization and Augmentation
    # 0.01 for AdamW
    # 0.0005 for SGD
    weight_decay: float = field(default=0.01)
    norm_weight_decay: float = field(default=0.0)
    gradient_max_norm: float = field(default=2.0)
    label_smoothing: float = 0.0

    # EMA configuration
    model_ema: bool = field(default=True)
    model_ema_steps: int = field(default=32)
    model_ema_decay: float = field(default=0.99998)

    mixed_precision: str = field(default="bf16")  # no for float32

    checkpoint_total_limit: int = field(default=3)
    checkpoint_epochs: int = field(default=1)
    save_image_epochs: int = field(default=1)
    seed: int = field(default=0)

    num_workers: int = field(default=2)

    # eval
    box_min_size: float = field(default=5.0)
    box_min_area: float = field(default=50.0)
    eval_epochs: int = field(default=1)

    # Matcher configuration
    # Class coefficient in the matching loss
    set_cost_class: float = field(default=1)
    # L1 box coefficient in the matching cost
    set_cost_bbox: float = field(default=5)
    # GIOU box coefficient in the matching cost
    set_cost_giou: float = field(default=2)

    # Loss coefficients
    bbox_loss_coef: float = field(default=5)
    giou_loss_coef: float = field(default=2)
    # Relative classification weight of the no-object class
    eos_coef: float = field(default=0.1)
