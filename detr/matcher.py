"""
HungarianMatcher implementation adapted from
https://github.com/facebookresearch/detr/blob/main/models/matcher.py
"""

from typing import List
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.transforms.v2.functional import convert_bounding_box_format

from detr.utils import generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Creates the matcher.

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost.
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost.
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(
        self,
        batch_pred_logits: torch.Tensor,
        batch_pred_boxes: torch.Tensor,
        batch_gt_labels: List[torch.Tensor],
        batch_gt_boxes: List[torch.Tensor],
    ):
        """
        Performs the matching.

        Params:
            batch_pred_logits: Tensor of dim [batch_size, num_queries, num_classes] with the classification logits.
            batch_pred_boxes: Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            batch_gt_labels: List of Tensors of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                             objects in the image) containing the class labels.
            batch_gt_boxes: List of Tensors of dim [num_target_boxes, 4] containing the target box coordinates.
                            gt_boxes are normalized to [0,1].

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order).
                - index_j is the indices of the corresponding selected gt box (in order).
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        batch_pred_probs = batch_pred_logits.softmax(-1)

        assignments = []
        for pred_probs, pred_boxes, gt_labels, gt_boxes in zip(
            batch_pred_probs, batch_pred_boxes, batch_gt_labels, batch_gt_boxes
        ):
            # pred_probs shape [num_queries, num_classes]
            # pred_boxes shape [num_queries, 4]
            # gt_labels shape [num_gt_boxes]
            # gt_boxes shape [num_gt_boxes, 4]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, gt_labels]

            # Compute the L1 cost between boxes
            # each box is represented by 4 values (center_x, center_y, w, h), each in range [0, 1]
            # so the maximum cost value between two boxes is 4
            gt_boxes_cxcywh = convert_bounding_box_format(gt_boxes, BoundingBoxFormat.XYXY, BoundingBoxFormat.CXCYWH)
            cost_bbox = torch.cdist(pred_boxes, gt_boxes_cxcywh, p=1)

            # GIOU in range (-1, 1]. cost_giou = -GIOU [-1, 1), typically GIOU_loss = 1 - GIOU in range [0, 2).
            pred_boxes_xyxy = convert_bounding_box_format(pred_boxes, BoundingBoxFormat.CXCYWH, BoundingBoxFormat.XYXY)
            cost_giou = -generalized_box_iou(pred_boxes_xyxy, gt_boxes)

            # shape [num_queries, num_gt_boxes]
            cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            indices = linear_sum_assignment(cost_matrix.cpu())
            pred_indices = torch.as_tensor(indices[0], dtype=torch.long)
            gt_indices = torch.as_tensor(indices[1], dtype=torch.long)
            assignments.append((pred_indices, gt_indices))

        return assignments
