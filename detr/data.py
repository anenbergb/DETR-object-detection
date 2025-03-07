from typing import Optional, Callable, Any, Dict, List, Tuple
import os
import torch
from torchvision.transforms import v2
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

from torchvision import tv_tensors
from torchvision.transforms.v2._utils import _get_fill


def labels_getter(inputs: Any) -> List[torch.Tensor]:
    # targets dictionary should be the second element in the tuple
    if isinstance(inputs, (tuple, list)):
        inputs = inputs[1]

    label_names = ["class_idx", "class_id", "iscrowd"]
    return [inputs[label_name] for label_name in label_names]


def get_train_transforms(
    resize_size=608,
    mean=(0.485, 0.456, 0.406),  # ImageNet mean and std
    std=(0.229, 0.224, 0.225),
):
    return v2.Compose(
        [
            v2.ToImage(),  # convert PIL image to tensor
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomPhotometricDistort(
                brightness=(0.875, 1.125),
                contrast=(0.7, 1.3),
                saturation=(0.8, 1.2),
                hue=(-0.05, 0.05),
                p=0.5,
            ),
            v2.Resize((resize_size, resize_size)),  # resize the image. bilinear
            v2.ClampBoundingBoxes(),  # clamp bounding boxes to be within the image
            v2.SanitizeBoundingBoxes(
                labels_getter=labels_getter
            ),  # Remove degenerate/invalid bounding boxes and their corresponding labels and masks.
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
            # v2.ToPureTensor(),
        ]
    )


# Other possible transforms to consider
# https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py
def get_val_transforms(
    resize_size=608,
    mean=(0.485, 0.456, 0.406),  # ImageNet mean and std
    std=(0.229, 0.224, 0.225),
):
    return v2.Compose(
        [
            v2.ToImage(),  # convert PIL image to tensor
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((resize_size, resize_size)),  # resize the image. bilinear
            v2.ClampBoundingBoxes(),  # clamp bounding boxes to be within the image
            v2.SanitizeBoundingBoxes(
                labels_getter=labels_getter
            ),  # Remove degenerate/invalid bounding boxes and their corresponding labels and masks.
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
            # v2.ToPureTensor(),
        ]
    )


class CocoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        assert split in ("train", "validation")
        root = os.path.join(dataset_root, split, "data")
        annFile = os.path.join(dataset_root, split, "labels.json")
        # https://pytorch.org/vision/main/generated/torchvision.datasets.wrap_dataset_for_transforms_v2.html#torchvision.datasets.wrap_dataset_for_transforms_v2
        self.dataset = wrap_dataset_for_transforms_v2(
            CocoDetection(root=root, annFile=annFile),
            target_keys=["image_id", "boxes", "labels", "iscrowd"],
        )
        self.transform = transform

        self.class_id2name = {id: d["name"] for id, d in self.dataset.coco.cats.items()}
        self.class_id2idx = {id: idx for idx, id in enumerate(self.dataset.coco.cats.keys())}

        self.class_names = [d["name"] for d in self.dataset.coco.cats.values()]
        self.class_idx2id = {idx: id for id, idx in self.class_id2idx.items()}

    @property
    def num_classes(self):
        return len(self.dataset.coco.cats)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        target contains:
            "image_id": int
            "boxes": torch.tensor (N,4)
            "labels": torch.tensor (N,)

        updates the target to include:
            "image_id": int
            "boxes": torch.tensor (N,4)
            "class_name": List[str]
            "class_idx": torch.tensor (N,)
            "class_id": torch.tensor (N,)
            "iscrows": torch.tensor (N,)
        """
        img, target = self.dataset[idx]
        if "labels" not in target:  # empty image
            target["boxes"] = tv_tensors.BoundingBoxes(
                torch.empty(0, 4, dtype=torch.float),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(img.height, img.width),
            )
            target["labels"] = torch.tensor([], dtype=torch.long)
            target["iscrowd"] = []

        class_ids = target.pop("labels")
        # target["class_name"] = [self.class_id2name[id.item()] for id in class_ids]
        target["class_idx"] = torch.tensor([self.class_id2idx[id.item()] for id in class_ids])
        target["class_id"] = class_ids
        target["iscrowd"] = torch.tensor(target["iscrowd"], dtype=torch.bool)
        if self.transform:
            img, target = self.transform(img, target)
        return img, target

class PadToMultipleOf32(v2.Pad):
    def __init__(self, fill=0, padding_mode="constant"):
        super().__init__(padding=0, fill=fill, padding_mode=padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # bounding boxes don't need padding since padding is applied
        # to the bottom and right sides
        padding = 0
        if isinstance(inpt, tv_tensors.Image):
            height, width = inpt.shape[-2:]
            pad_width = (32 - width % 32) % 32
            pad_height = (32 - height % 32) % 32
            padding = (0, 0, pad_width, pad_height)
        fill = _get_fill(self._fill, type(inpt))
        return self._call_kernel(
            v2.functional.pad,
            inpt,
            padding=padding,
            fill=fill,
            padding_mode=self.padding_mode,
        )
