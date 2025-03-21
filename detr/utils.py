from typing import List
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops.boxes import box_area
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.transforms.v2.functional import convert_bounding_box_format


class DetectionMetrics:
    """
    https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
    """

    def __init__(self, class_names: List[str], backend: str = "pycocotools"):
        # Use the default IOU thresholds of [0.5,...,0.95] with step 0.01
        self.metric = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", class_metrics=True, backend=backend, sync_on_compute=False
        )
        self.class_names = class_names

    def update(self, preds, batch):
        """
        batch is a dictionary of tensors with keys
        {"image", "image_id", "boxes", "class_id", "iscrowd", etc.}

        preds is a list of dictionaries, each dictionary cooresponding to a batch item
        with keys {"boxes", "scores", "labels", etc}

        boxes must be XYXY format
        """
        targets = [
            {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}
            for boxes, labels, iscrowd in zip(batch["boxes"], batch["class_idx"], batch["iscrowd"])
        ]
        self.metric.update(preds, targets)

    def compute(self):
        output = self.metric.compute()
        metrics = {
            "AP": output["map"].item(),
            "AP50": output["map_50"].item(),
            "AP75": output["map_75"].item(),
            "AP-large": output["map_large"].item(),
            "AP-medium": output["map_medium"].item(),
            "AP-small": output["map_small"].item(),
        }
        map_per_class = torch.zeros(len(self.class_names), dtype=torch.float)
        map_per_class[output["classes"]] = output["map_per_class"]
        for class_name, map_score in zip(self.class_names, map_per_class):
            metrics[f"AP-per-class/{class_name}"] = map_score.item()
        return metrics

    def reset(self):
        self.metric.reset()


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)

    source implementation
    https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class PostProcess(torch.nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, class_names: List[str]):
        super().__init__()
        self.class_names = class_names

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, image_heights, image_widths, score_threshold=0.0):
        """
        pred_logits: Tensor of dim [batch_size, num_queries, num_classes] with the classification logits.
        pred_boxes: Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.

        image_heights: Tensor of dim [batch_size] with the image heights.
        image_widths: Tensor of dim [batch_size] with the image widths.

        score_threshold: float, threshold used to filter detections with low confidence.
        """
        assert len(pred_logits) == len(pred_boxes)

        pred_probs = pred_logits.softmax(-1)
        # exclude the probability for no object as an option for the highest class
        scores, labels = pred_probs[..., :-1].max(-1)
        pred_boxes_xyxy = convert_bounding_box_format(pred_boxes, BoundingBoxFormat.CXCYWH, BoundingBoxFormat.XYXY)
        scale_fct = torch.stack([image_widths, image_heights, image_widths, image_heights], dim=1)
        pred_boxes_xyxy_scaled = pred_boxes_xyxy * scale_fct[:, None, :]

        results = []
        for s, label_indices, b in zip(scores, labels, pred_boxes_xyxy_scaled):
            # Apply score threshold
            keep = s >= score_threshold
            s = s[keep]
            label_indices = label_indices[keep]
            b = b[keep]
            class_names = [self.class_names[i] for i in label_indices]
            results.append({"scores": s, "labels": label_indices, "boxes": b, "class_names": class_names})
        return results
