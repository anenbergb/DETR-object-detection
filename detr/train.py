import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple
import logging
from collections import defaultdict

# Huggingface
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object
from safetensors.torch import load_model

from detr.model import DETR, DETRConfig
from detr.data import (
    CocoDataset,
    get_val_transforms,
    get_train_transforms,
    get_collate_function,
)
from detr.matcher import HungarianMatcher
from detr.loss import SetCriterion
from detr.utils import DetectionMetrics, PostProcess
from detr.visualize import plot_grid


@dataclass
class TrainingConfig:
    output_dir: str
    overwrite_output_dir: bool = field(default=True)  # overwrite the old model
    start_epoch: int = field(default=0)
    resume_from_checkpoint: str = field(default=None)

    coco_dataset_root: str = field(default="/media/bryan/ssd01/fiftyone/coco-2017")

    # the train batch size used in the paper when training across multiple GPUs
    cumulative_train_batch_size: int = field(default=64)
    train_batch_size: int = field(default=5)
    val_batch_size: int = field(default=25)

    epochs: int = field(default=100)
    limit_train_iters: int = field(default=0)
    limit_val_iters: int = field(default=0)

    # official DETR learning rate schedule is
    # 300 epochs at 1e-4, and final 100 epochs at 1e-5
    # we use linear warmup + cosine one-cycle decay to speed
    # up convergence.
    # 1e-4 for AdamW for post-LN Transformer
    # 3e-4 is possible for pre-LN Transformer due to improved stability
    lr: float = field(default=3e-4)
    lr_backbone: float = field(default=3e-5)  # official DETR uses 1e-5
    lr_warmup_epochs: int = field(default=5)
    lr_warmup_decay: float = field(default=0.01)
    # Maintain the maximum learning rate for an extended number of epochs
    # because DETR’s unique loss and query mechanism likely need more time
    # at LR_max than a typical Transformer classification task, even with pre-LN.
    lr_hold_max_epochs: int = field(default=35)
    # The lr_min is > 0 to minimize the risk of stagntation during the later
    # training stages when the learning rate is very low.
    lr_min: float = field(default=3e-6)

    # Regularization and Augmentation
    weight_decay: float = field(default=1e-4)
    norm_weight_decay: float = field(default=0.0)
    # official DETR implementation uses gradient clipping max norm 0.1,
    # which is fairly small.
    gradient_max_norm: float = field(default=1.0)

    # label_smoothing: float = 0.0

    mixed_precision: str = field(default="bf16")  # no for float32

    checkpoint_total_limit: int = field(default=3)
    checkpoint_epochs: int = field(default=1)
    save_image_epochs: int = field(default=1)
    seed: int = field(default=0)
    log_frequency: int = field(default=100)

    num_workers: int = field(default=2)

    # eval
    box_min_size: float = field(default=5.0)
    box_min_area: float = field(default=50.0)
    eval_epochs: int = field(default=10)

    # Matcher configuration
    # Class coefficient in the matching loss
    set_cost_class: float = field(default=1)
    # L1 box coefficient in the matching cost
    set_cost_bbox: float = field(default=5)
    # GIOU box coefficient in the matching cost
    set_cost_giou: float = field(default=2)

    # Loss coefficients
    label_ce_loss_coef: float = field(default=1)
    bbox_loss_coef: float = field(default=5)
    giou_loss_coef: float = field(default=2)
    # Relative classification weight of the no-object class
    eos_coef: float = field(default=0.1)


def train_DETR(config: TrainingConfig, detr_config: DETRConfig):

    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        # logging_dir
        automatic_checkpoint_naming=True,
        total_limit=config.checkpoint_total_limit,
        save_on_each_node=False,
        iteration=config.start_epoch,  # the current save iteration
    )
    gradient_accumulation_steps = config.cumulative_train_batch_size // config.train_batch_size
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
        step_scheduler_with_optimizer=False,
        split_batches=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(os.path.basename(config.output_dir))
        accelerator.print(
            f"Gradient Accumulation steps: {gradient_accumulation_steps}. "
            f"To achieve a cumulative batch size of {config.cumulative_train_batch_size} "
            f"given a per-GPU batch size of {config.train_batch_size}"
        )

    # logger = get_logger(__name__, log_level="DEBUG")
    # logger.info("INFO LEVEL", main_process_only=True)
    # logger.debug("DEBUG LEVEL", main_process_only=True)

    train_dataset = CocoDataset(
        dataset_root=config.coco_dataset_root,
        split="train",
        transform=get_train_transforms(),
    )
    val_dataset = CocoDataset(
        dataset_root=config.coco_dataset_root,
        split="validation",
        transform=get_val_transforms(),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=get_collate_function(),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config.num_workers,
        collate_fn=get_collate_function(),
    )
    detr_config.num_classes = train_dataset.num_classes

    model = DETR(detr_config)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"number of params: {n_parameters}")
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,  # ResNet-50 backbone with lower initial learning rate
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": config.lr,  # Transformer with higher initial learning rate
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay)

    for i, param_group in enumerate(optimizer.param_groups):
        accelerator.print(f"Parameter group {i}, initial learning rate: {param_group['lr']}")

    # 1. Warmup scheduler: Linear increase to max learning rate
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.lr_warmup_decay,
        total_iters=config.lr_warmup_epochs,
    )
    # 2. Constant LR scheduler: Keep learning rate fixed
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 1.0  # Multiplier of 1.0 to keep the LR constant
    )
    # 3. Cosine annealing scheduler: Gradually decrease learning rate
    cooldown_epochs = config.epochs - config.lr_warmup_epochs - config.lr_hold_max_epochs
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cooldown_epochs, eta_min=config.lr_min)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2, scheduler3],
        milestones=[config.lr_warmup_epochs, config.lr_warmup_epochs + config.lr_hold_max_epochs],
    )

    matcher = HungarianMatcher(
        cost_class=config.set_cost_class, cost_bbox=config.set_cost_bbox, cost_giou=config.set_cost_giou
    )
    criterion = SetCriterion(
        train_dataset.num_classes,
        matcher,
        weight_label_ce=config.label_ce_loss_coef,
        weight_bbox_l1=config.bbox_loss_coef,
        weight_bbox_giou=config.giou_loss_coef,
        eos_coef=config.eos_coef,
    )

    model, optimizer, criterion, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, criterion, train_dataloader, val_dataloader, scheduler
    )

    # ONLY load the model weights from the checkpoint. Leave the optimizer and scheduler as is.
    if config.resume_from_checkpoint is not None and os.path.exists(config.resume_from_checkpoint):
        model_fpath = os.path.join(config.resume_from_checkpoint, "model.safetensors")
        assert os.path.exists(model_fpath), f"Model file {model_fpath} not found"
        accelerator.print(f"Loading model weights from {model_fpath}")
        weights_before = model.input_proj.weight.detach().clone()
        load_model(
            accelerator._models[0],
            model_fpath,
            device=str(accelerator.device),
        )
        weight_after = model.input_proj.weight.detach().clone()
        assert not torch.allclose(
            weights_before, weight_after
        ), "Model weights did not change after loading from checkpoint"

    if config.start_epoch > 0:
        accelerator.print(f"Resuming training from epoch {config.start_epoch}")
        for _ in range(config.start_epoch):
            scheduler.step()

    global_step = 0
    for epoch in range(config.start_epoch, config.epochs):
        model.train()
        criterion.train()
        for step, batch in (
            progress_bar := tqdm(
                enumerate(train_dataloader),
                total=(len(train_dataloader) if config.limit_train_iters == 0 else config.limit_train_iters),
                disable=not accelerator.is_local_main_process,
                desc=f"Epoch {epoch}",
            )
        ):
            if config.limit_train_iters > 0 and step >= config.limit_train_iters:
                break

            with accelerator.accumulate(model):  # accumulate gradients into model.grad attributes
                with accelerator.autocast():
                    outputs = model(batch["image"], batch["height"], batch["width"])
                    loss_dict = criterion(outputs, batch)
                    loss = sum(v for k, v in loss_dict.items() if k.startswith("loss"))
                accelerator.backward(loss)  # accumulates gradients

            accelerator.clip_grad_norm_(model.parameters(), config.gradient_max_norm)
            optimizer.step()
            optimizer.zero_grad()

            backbone_lr, transformer_lr = scheduler.get_last_lr()
            logs = {
                "loss": loss.detach().item(),
                "lr/backbone": backbone_lr,
                "lr/transformer": transformer_lr,
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            if step % config.log_frequency == 0:
                logs["loss"] = {"train": loss.detach().item()}
                for k in loss_dict.keys():
                    loss_dict[k] = loss_dict[k].detach().item()
                logs.update(format_loss_for_logging(loss_dict, split="train"))
                accelerator.log(logs, step=global_step)
            global_step += 1

        if epoch % config.checkpoint_epochs == 0:
            accelerator.save_state()

        scheduler.step()  # once per epoch
        if epoch % config.eval_epochs == 0 or epoch == config.epochs - 1:
            val_metrics = run_validation(
                accelerator,
                model,
                criterion,
                val_dataloader,
                limit_val_iters=config.limit_val_iters,
                global_step=global_step,
            )
            if accelerator.is_main_process:
                val_print_str = f"Validation metrics [Epoch {epoch}]: "
                AP = val_metrics.pop("AP", 0.0)
                AP50 = val_metrics.pop("AP50", 0.0)
                AP_person = val_metrics.get("AP-per-class/person", 0.0)
                AP_cat = val_metrics.get("AP-per-class/cat", 0.0)
                val_print_str += f"AP: {AP:.3f} AP50: {AP50:.3f} AP-person: {AP_person:.3f} AP-cat: {AP_cat:.3f}"
                accelerator.print(val_print_str)

                # consolidate AP plots
                AP75 = val_metrics.pop("AP75", 0.0)
                val_metrics["Average Precision"] = {
                    "AP": AP,
                    "AP50": AP50,
                    "AP75": AP75,
                }
                AP_large = val_metrics.pop("AP-large", 0.0)
                AP_medium = val_metrics.pop("AP-medium", 0.0)
                AP_small = val_metrics.pop("AP-small", 0.0)
                val_metrics["Average Precision by Object Size"] = {
                    "AP-large": AP_large,
                    "AP-medium": AP_medium,
                    "AP-small": AP_small,
                }
                accelerator.log(val_metrics, step=global_step)

    accelerator.end_training()


def format_loss_for_logging(loss_dict, split="train"):
    logs = {}
    loss_names = [
        "loss_label_ce",
        "loss_l1_bbox",
        "loss_giou",
    ]
    loss_prefix_len = len("loss_")
    for loss_name in loss_names:
        name = loss_name[loss_prefix_len:]
        logs[f"loss/{name}"] = {split: loss_dict[loss_name]}

        loss_by_layer = {}
        for k, v in loss_dict.items():
            if k.startswith(loss_name):
                loss_by_layer[k] = v

        logs[f"loss by decoder layer/{name}-{split}"] = loss_by_layer

    name = "Incorrect Number of Object Predictions"
    logs[f"{name}/cardinality_error"] = {split: loss_dict["cardinality_error"]}
    error_by_layer = {}
    for k, v in loss_dict.items():
        if k.startswith("cardinality_error"):
            error_by_layer[k] = v
    logs[f"{name}/by decoder layer {split}"] = error_by_layer

    logs["Classification Error (1 - accuracy)"] = {split: loss_dict["class_error"]}
    return logs


def run_validation(
    accelerator,
    model,
    criterion,
    val_dataloader,
    limit_val_iters=0,
    global_step=0,
):
    """
    NOTE: This function is written without consideration for distributed multi-GPU training.
    """
    post_process = PostProcess(val_dataloader.dataset.class_names)
    metrics = DetectionMetrics(val_dataloader.dataset.class_names)
    total_num_images = len(val_dataloader.dataset)
    avg_loss_dict = defaultdict(float)

    model.eval()
    criterion.eval()
    with torch.inference_mode():
        for step, batch in tqdm(
            enumerate(val_dataloader),
            total=len(val_dataloader) if limit_val_iters == 0 else limit_val_iters,
            disable=not accelerator.is_local_main_process,
            desc="Validation",
        ):
            if limit_val_iters > 0 and step >= limit_val_iters:
                break

            with accelerator.autocast():
                outputs = model(batch["image"], batch["height"], batch["width"])
                loss_dict = criterion(outputs, batch)

            num_images = batch["image"].size(0)
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.detach().item() * num_images / total_num_images

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].detach().cpu()
                elif isinstance(batch[key], list):
                    if isinstance(batch[key][0], torch.Tensor):
                        batch[key] = [x.detach().cpu() for x in batch[key]]

            preds = post_process(
                outputs["pred_logits"][:, -1].detach().cpu(),
                outputs["pred_boxes"][:, -1].detach().cpu(),
                batch["height"].detach().cpu(),
                batch["width"].detach().cpu(),
            )
            metrics.update(preds, batch)

            # log the predictions for the first batch
            # Accelerate tensorboard tracker
            # https://github.com/huggingface/accelerate/blob/main/src/accelerate/tracking.py#L165
            if step == 0:
                batch_flat = []
                for i in range(batch["image"].shape[0]):
                    item = {
                        "image": batch["image"][i].detach().cpu(),
                        "boxes": batch["boxes"][i].detach().cpu(),
                        "class_names": [val_dataloader.dataset.class_names[c] for c in batch["class_idx"][i].tolist()],
                    }
                    batch_flat.append(item)
                    preds[i]["image"] = item["image"]

                vis_gt = plot_grid(
                    batch_flat,
                    max_images=25,
                    num_cols=5,
                    font_size=20,
                    box_color="green",
                    fig_scaling=7,
                )
                vis_preds = plot_grid(
                    preds,
                    max_images=25,
                    num_cols=5,
                    font_size=20,
                    box_color="red",
                    fig_scaling=7,
                )

                tensorboard = accelerator.get_tracker("tensorboard")
                tensorboard.log_images(
                    {
                        "val-ground-truth": vis_gt,
                        "val-predictions": vis_preds,
                    },
                    step=global_step,
                    dataformats="HWC",
                )

    avg_loss = sum([loss for k, loss in avg_loss_dict.items() if k.startswith("loss")])
    logs = {"loss": {"val": avg_loss}}
    logs.update(format_loss_for_logging(loss_dict, split="val"))
    accelerator.log(logs, step=global_step)

    val_metrics = metrics.compute()

    torch.cuda.empty_cache()
    return val_metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Run training loop for YoloV3 object detection model on the COCO dataset.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/bryan/ssd01/expr/detr_from_scratch/run01",
        help="Path to save the model",
    )
    parser.add_argument(
        "--coco-dataset-root",
        type=str,
        default="/media/bryan/ssd01/fiftyone/coco-2017",
        help="Path to the COCO dataset",
    )
    parser.add_argument("--train-batch-size", type=int, default=5, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=25, help="Validation batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--lr-warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--lr-hold-max-epochs", type=int, default=35, help="Hold max LR epochs")
    parser.add_argument(
        "--limit-train-iters",
        type=int,
        default=0,
        help="Limit number of training iterations per epoch",
    )
    parser.add_argument(
        "--limit-val-iters",
        type=int,
        default=0,
        help="Limit number of val iterations per epoch",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Start epoch, useful for resuming training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint folder that the training should resume from.",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=10,
        help="Frequency of evaluation in epochs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig()
    args = get_args()
    config = TrainingConfig(
        output_dir=args.output_dir,
        coco_dataset_root=args.coco_dataset_root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_hold_max_epochs=args.lr_hold_max_epochs,
        limit_train_iters=args.limit_train_iters,
        limit_val_iters=args.limit_val_iters,
        start_epoch=args.start_epoch,
        resume_from_checkpoint=args.resume_from_checkpoint,
        eval_epochs=args.eval_epochs,
    )
    # optionally overwrite DETR model parameters
    detr_config = DETRConfig()
    sys.exit(train_DETR(config, detr_config))
