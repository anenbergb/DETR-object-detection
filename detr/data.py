from typing import Optional, Callable, Any, Dict, List, Tuple
import os
import torch
from torchvision.transforms import v2
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

from torchvision import tv_tensors
from torchvision.transforms.v2._utils import _get_fill, query_size


def labels_getter(inputs: Any) -> List[torch.Tensor]:
    # targets dictionary should be the second element in the tuple
    if isinstance(inputs, (tuple, list)):
        inputs = inputs[1]

    label_names = ["class_idx", "class_id", "iscrowd"]
    return [inputs[label_name] for label_name in label_names]


def get_train_transforms(
    mean=(0.485, 0.456, 0.406),  # ImageNet mean and std
    std=(0.229, 0.224, 0.225),
):
    """
    Random crops transforms are in the style of official detr implementation
    https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
    """
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333

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
            v2.RandomChoice(
                [
                    v2.RandomShortestSize(scales, max_size),  # bilinear
                    v2.Compose(
                        [
                            v2.RandomShortestSize([400, 500, 600]),  # bilinear
                            RandomSizeCrop(384, 600),
                            v2.RandomShortestSize(scales, max_size),  # bilinear
                        ]
                    ),
                ],
            ),
            v2.ClampBoundingBoxes(),  # clamp bounding boxes to be within the image
            v2.SanitizeBoundingBoxes(
                labels_getter=labels_getter
            ),  # Remove degenerate/invalid bounding boxes and their corresponding labels and masks.
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
            # v2.ToPureTensor(),
        ]
    )


def get_val_transforms(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), min_size=800, max_size=1333  # ImageNet mean and std
):
    return v2.Compose(
        [
            v2.ToImage(),  # convert PIL image to tensor
            v2.ToDtype(torch.uint8, scale=True),
            # maintain aspect ratio, resize the image to have smallest size = min_size
            v2.RandomShortestSize(min_size, max_size),  # bilinear
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
            "boxes": torch.tensor (N,4) XYXY format
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


class RandomSizeCrop(v2.RandomCrop):
    def __init__(
        self,
        min_size: int,
        max_size: int,
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
        super().__init__((100, 100), padding=None, pad_if_needed=False)

    # The crop should be inbounds, no padding, so don't need to initialize any of those arguments to RandomCrop parent class
    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        image_height, image_width = query_size(flat_inputs)
        cropped_width = torch.randint(self.min_size, min(image_width, self.max_size), size=(1,)).item()
        cropped_height = torch.randint(self.min_size, min(image_height, self.max_size), size=(1,)).item()

        top = torch.randint(0, image_height - cropped_height + 1, size=()).item()
        left = torch.randint(0, image_width - cropped_width + 1, size=()).item()

        return dict(
            needs_crop=True,
            top=top,
            left=left,
            height=cropped_height,
            width=cropped_width,
            needs_pad=False,
            padding=[0, 0, 0, 0],
        )


def max_by_axis(the_list: List[list[int]], divisible_by: int = 32):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def get_collate_function(divisible_by: int = 32):
    def collate_function(batched_image_target):
        images, targets = list(zip(*batched_image_target))
        # each image is of shape (C, H, W)
        heights = torch.tensor([img.shape[1] for img in images], dtype=torch.int)
        widths = torch.tensor([img.shape[2] for img in images], dtype=torch.int)

        max_size = max_by_axis([list(img.shape) for img in images])
        # ensure that the height and width are divisible by 32
        max_size[1:] = [((x + divisible_by - 1) // divisible_by) * divisible_by for x in max_size[1:]]
        batch_shape = [len(images)] + max_size
        dtype = images[0].dtype
        batch_tensor = torch.zeros(batch_shape, dtype=dtype)
        for img, pad_img in zip(images, batch_tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        batch = {
            "image": batch_tensor,
            "height": heights,
            "width": widths,
            "image_id": torch.tensor([target["image_id"] for target in targets], dtype=torch.int),
        }

        label_names = ["boxes", "class_idx", "class_id", "iscrowd"]
        for label_name in label_names:
            batch[label_name] = [x[label_name] for x in targets]

        batch["boxes_normalized"] = []
        for height, width, boxes in zip(batch["height"], batch["width"], batch["boxes"]):
            copied = boxes.clone()
            copied[..., [0, 2]] /= width
            copied[..., [1, 3]] /= height
            batch["boxes_normalized"].append(copied)

        return batch

    return collate_function
