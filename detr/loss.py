from typing import List, Tuple
import torch
from torch import nn

from detr.utils import accuracy


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
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
        Classification loss (NLL)

        pred_logits shape [batch_size, num_queries, num_classes + 1]
        gt_labels shape List[ [num_gt_boxes] ]

        indices
        A list of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order).
            - index_j is the indices of the corresponding selected gt box (in order).

        """

        # batch_idx of shape (total_num_gt_boxes,), src_idx of shape (total_num_gt_boxes,)
        # e.g. batch_idx = [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
        # e.g. src_idx = [1,6,8,, ...]
        idx = self._get_src_permutation_idx(indices)

        # (total_num_gt_boxes,)
        target_classes_o = torch.cat(
            [gt_labels[gt_box_indices] for gt_labels, (_, gt_box_indices) in zip(batch_gt_labels, indices)]
        )

        # (batch_size, num_queries)
        # fill the target_classes matrix with value equal to self.num_classes,
        # which is the background no-object class
        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device
        )
        # make the assignment of the correct ground truth class label to the predicted class label
        target_classes[idx] = target_classes_o

        pred_logits = pred_logits.flatten(0, 1)  # (batch_size * num_queries, num_classes + 1)
        target_classes = target_classes.flatten()  # (batch_size * num_queries,)

        loss = torch.nn.functional.cross_entropy(pred_logits, target_classes, self.empty_weight)
        if include_class_error:
            class_error = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
            return loss, class_error
        return loss

    # @torch.no_grad()
    # def loss_cardinality(self, outputs, targets, indices, num_boxes):
    #     """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
    #     This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
    #     """
    #     pred_logits = outputs["pred_logits"]
    #     device = pred_logits.device
    #     tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
    #     # Count the number of predictions that are NOT "no-object" (which is the last class)
    #     card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
    #     card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    #     losses = {"cardinality_error": card_err}
    #     return losses

    # def loss_boxes(self, outputs, targets, indices, num_boxes):
    #     """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
    #     targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
    #     The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    #     """
    #     assert "pred_boxes" in outputs
    #     idx = self._get_src_permutation_idx(indices)
    #     src_boxes = outputs["pred_boxes"][idx]
    #     target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

    #     loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

    #     losses = {}
    #     losses["loss_bbox"] = loss_bbox.sum() / num_boxes

    #     loss_giou = 1 - torch.diag(
    #         box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
    #     )
    #     losses["loss_giou"] = loss_giou.sum() / num_boxes
    #     return losses

    # def loss_masks(self, outputs, targets, indices, num_boxes):
    #     """Compute the losses related to the masks: the focal loss and the dice loss.
    #     targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    #     """
    #     assert "pred_masks" in outputs

    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)
    #     src_masks = outputs["pred_masks"]
    #     src_masks = src_masks[src_idx]
    #     masks = [t["masks"] for t in targets]
    #     # TODO use valid to mask invalid areas due to padding in loss
    #     target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    #     target_masks = target_masks.to(src_masks)
    #     target_masks = target_masks[tgt_idx]

    #     # upsample predictions to the target size
    #     src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
    #     src_masks = src_masks[:, 0].flatten(1)

    #     target_masks = target_masks.flatten(1)
    #     target_masks = target_masks.view(src_masks.shape)
    #     losses = {
    #         "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
    #         "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
    #     }
    #     return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
        """

        losses = {}
        layerwise_pred_logits = outputs["pred_logits"]
        layerwise_pred_boxes = outputs["pred_boxes"]
        num_layers = layerwise_pred_logits.size(1)  # should be 6
        for i in range(num_layers):
            pred_logits = layerwise_pred_logits[:, i]
            pred_boxes = layerwise_pred_boxes[:, i]

            # Retrieve the matching between the outputs of the last layer and the targets
            # pred_boxes are expected in (center_x, center_y, w, h) format, normalized to range [0, 1]
            # boxes_normalized are expected in format XYXY, normalized to range [0, 1]
            indices = self.matcher(pred_logits, pred_boxes, targets["class_idx"], targets["boxes_normalized"])

            num_boxes_across_batch = sum(len(x) for x in targets["class_idx"])

            # Compute all the requested losses
            loss_suffix = f"_{i}" if i < num_layers - 1 else ""
            loss_labels = self.loss_labels(
                pred_logits, targets["class_idx"], indices, include_class_error=i == num_layers - 1
            )
            if i == num_layers - 1:
                losses["class_error"] = loss_labels[1]
                loss_labels = loss_labels[0]
            losses[f"loss_ce{loss_suffix}"] = loss_labels

            # loss_cardinality
            # loss_boxes

        return losses
