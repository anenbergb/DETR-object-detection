"""
SetCriterion implementation adapted from
https://github.com/facebookresearch/detr/blob/main/models/detr.py
"""

from typing import List, Tuple
import torch
from torch import nn

from detr.matcher import HungarianMatcher
from detr.utils import accuracy, generalized_box_iou
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.transforms.v2.functional import convert_bounding_box_format


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.

    The process happens in two steps:
        1) Compute Hungarian assignment between ground truth boxes and the outputs of the model.
        2) Supervise each pair of matched ground-truth / prediction (supervise class and box).
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_label_ce: float = 1.0,
        weight_bbox_l1: float = 5.0,
        weight_bbox_giou: float = 2.0,
        eos_coef=0.1,
    ):
        """
        Create the criterion.

        Parameters:
            num_classes: Number of object categories, omitting the special no-object category.
            matcher: Module able to compute a matching between targets and proposals.
            weight_label_ce: Relative weight of the classification error in the loss.
            weight_bbox_l1: Relative weight of the L1 error of the bounding box coordinates in the loss.
            weight_bbox_giou: Relative weight of the GIoU loss of the bounding box in the loss.
            eos_coef: Relative classification weight applied to the no-object category.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_label_ce = weight_label_ce
        self.weight_bbox_l1 = weight_bbox_l1
        self.weight_bbox_giou = weight_bbox_giou
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(
        self,
        pred_logits: torch.Tensor,
        batch_gt_labels: List[torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        include_class_error: bool = False,
    ):
        """
        Compute the classification loss (NLL).

        Parameters:
            pred_logits: Tensor of shape [batch_size, num_queries, num_classes + 1] with the classification logits.
            batch_gt_labels: List of Tensors of shape [num_gt_boxes] containing the class labels for each ground-truth box.
            indices: List of tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order).
                - index_j is the indices of the corresponding selected ground-truth box (in order).
            include_class_error: Boolean flag to include classification error in the output.

        Returns:
            The classification loss. If include_class_error is True, also returns the classification error.
        """
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [gt_labels[gt_box_indices] for gt_labels, (_, gt_box_indices) in zip(batch_gt_labels, indices)]
        )
        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o

        pred_logits_flat = pred_logits.flatten(0, 1)
        target_classes_flat = target_classes.flatten()

        loss = torch.nn.functional.cross_entropy(pred_logits_flat, target_classes_flat, self.empty_weight)
        loss *= self.weight_label_ce
        if include_class_error:
            class_error = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
            return loss, class_error
        return loss

    @torch.no_grad()
    def loss_cardinality(
        self,
        pred_logits: torch.Tensor,
        batch_gt_labels: List[torch.Tensor],
    ):
        """
        Compute the cardinality error, i.e., the absolute error in the number of predicted non-empty boxes.
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.

        Parameters:
            pred_logits: Tensor of shape [batch_size, num_queries, num_classes + 1] with the classification logits.
            batch_gt_labels: List of Tensors of shape [num_gt_boxes] containing the class labels for each ground-truth box.

        Returns:
            The cardinality error.
        """
        device = pred_logits.device
        num_gt_boxes = torch.as_tensor(
            [len(gt_labels) for gt_labels in batch_gt_labels], device=device, dtype=torch.float
        )
        no_object_class = pred_logits.shape[-1] - 1
        pred_class = pred_logits.argmax(-1)  # [batch_size, num_queries]
        card_pred = (pred_class != no_object_class).sum(1).float()  # [batch_size]
        return torch.nn.functional.l1_loss(card_pred, num_gt_boxes)

    def loss_boxes(
        self,
        pred_boxes: torch.Tensor,
        batch_gt_boxes: List[torch.Tensor],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Parameters:
            pred_boxes: Tensor of shape [batch_size, num_queries, 4] with the predicted box coordinates.
            batch_gt_boxes: List of Tensors of shape [num_gt_boxes, 4] containing the ground-truth box coordinates.
            indices: List of tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order).
                - index_j is the indices of the corresponding selected ground-truth box (in order).

        Returns:
            The L1 loss and the GIoU loss.
        """
        total_num_gt_boxes = max(sum([len(gt_boxes) for gt_boxes in batch_gt_boxes]), 1)
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat(
            [gt_boxes[gt_box_indices] for gt_boxes, (_, gt_box_indices) in zip(batch_gt_boxes, indices)], dim=0
        )

        target_boxes_cxcywh = convert_bounding_box_format(
            target_boxes, BoundingBoxFormat.XYXY, BoundingBoxFormat.CXCYWH
        )
        loss_bbox = torch.nn.functional.l1_loss(src_boxes, target_boxes_cxcywh, reduction="sum") / total_num_gt_boxes
        loss_bbox *= self.weight_bbox_l1

        src_boxes_xyxy = convert_bounding_box_format(src_boxes, BoundingBoxFormat.CXCYWH, BoundingBoxFormat.XYXY)
        giou = generalized_box_iou(src_boxes_xyxy, target_boxes)
        loss_giou = 1 - torch.diag(giou)
        loss_giou = loss_giou.sum() / total_num_gt_boxes
        loss_giou *= self.weight_bbox_giou

        return loss_bbox, loss_giou

    def _get_src_permutation_idx(self, indices):
        """
        Permute predictions following indices.

        Parameters:
            indices: List of tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order).
                - index_j is the indices of the corresponding selected ground-truth box (in order).

        Returns:
            The batch indices and source indices.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """
        Permute targets following indices.

        Parameters:
            indices: List of tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order).
                - index_j is the indices of the corresponding selected ground-truth box (in order).

        Returns:
            The batch indices and target indices.
        """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """
        Perform the loss computation.

        Parameters:
            outputs: Dict of tensors, see the output specification of the model for the format.
            targets: Dict containing the ground-truth information.

        Returns:
            A dictionary containing the computed losses.
        """
        losses = {}
        layerwise_pred_logits = outputs["pred_logits"]
        layerwise_pred_boxes = outputs["pred_boxes"]
        num_layers = layerwise_pred_logits.size(1)
        for i in range(num_layers):
            pred_logits = layerwise_pred_logits[:, i]
            pred_boxes = layerwise_pred_boxes[:, i]

            indices = self.matcher(pred_logits, pred_boxes, targets["class_idx"], targets["boxes_normalized"])

            loss_suffix = f"_{i}" if i < num_layers - 1 else ""
            loss_labels = self.loss_labels(
                pred_logits, targets["class_idx"], indices, include_class_error=i == num_layers - 1
            )
            if i == num_layers - 1:
                losses["class_error"] = loss_labels[1]
                loss_labels = loss_labels[0]
            losses[f"loss_label_ce{loss_suffix}"] = loss_labels
            losses[f"cardinality_error{loss_suffix}"] = self.loss_cardinality(pred_logits, targets["class_idx"])
            losses[f"loss_l1_bbox{loss_suffix}"], losses[f"loss_giou{loss_suffix}"] = self.loss_boxes(
                pred_boxes, targets["boxes_normalized"], indices
            )
        return losses
