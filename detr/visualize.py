from PIL import ImageDraw, ImageFont

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import itertools


def plot_grid(
    image_dicts,
    max_images=25,
    num_cols=5,
    **kwargs,
):
    image_target_tuples = [(x["image"], x) for x in image_dicts][:max_images]
    image_rows = [list(x) for x in itertools.batched(image_target_tuples, num_cols)]
    return plot(image_rows, **kwargs)


# adopted from https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py
def plot(
    imgs,
    row_title=None,
    box_color="green",
    box_width=3,
    font="FreeSans.ttf",
    font_size=20,
    fig_scaling=3,
    **imshow_kwargs,
):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig_scaling = fig_scaling
    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        squeeze=False,
        figsize=(fig_scaling * num_cols, fig_scaling * num_rows),
    )
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                    class_names = target.get("class_names")
                    scores = target.get("scores")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None and len(boxes) > 0:
                labels = None
                if class_names is not None and scores is not None:
                    labels = [f"{class_names[i]}: {scores[i]:.2f}" for i in range(len(class_names))]
                elif class_names is not None:
                    labels = class_names
                img = draw_bounding_boxes(
                    img,
                    boxes,
                    labels=labels,
                    colors=box_color,
                    fill=False,
                    width=box_width,
                    font=font,
                    font_size=font_size,
                    label_colors=box_color,
                )
            if masks is not None:
                img = draw_segmentation_masks(
                    img,
                    masks.to(torch.bool),
                    colors=["green"] * masks.shape[0],
                    alpha=0.65,
                )

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.axis("off")

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    # Adjust layout to reduce white space
    fig.tight_layout(pad=0)

    # Save the figure to a numpy array
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # Convert ARGB to RGB
    image_array = image_array[..., [1, 2, 3]]

    plt.close(fig)  # Close the figure to free memory
    return image_array


def render_bounding_boxes(image, annotations, category_names):
    """
    Render bounding boxes and category_id tags on an image.

    :param image: PIL Image.
    :param annotations: List of annotations, where each annotation is a dictionary
                        with keys 'bbox' and 'category_id'.
    :param category_names: Dictionary mapping category_id to category name.
    :return: PIL.Image object with bounding boxes and category_id tags.
    """
    # Load the image
    draw = ImageDraw.Draw(image)

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw each bounding box and category_id tag
    for annotation in annotations:
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        category_name = category_names.get(category_id, "Unknown")

        # Draw the bounding box
        x, y, width, height = bbox
        draw.rectangle(((x, y), (x + width, y + height)), outline="red", width=2)

        # Draw the category_id tag
        text = f"{category_name} ({category_id})"
        text_bbox = draw.textbbox((x, y), text, font=font)
        text_location = (x, y - (text_bbox[3] - text_bbox[1]))
        draw.rectangle(
            (
                (x, y - (text_bbox[3] - text_bbox[1])),
                (x + (text_bbox[2] - text_bbox[0]), y),
            ),
            fill="red",
        )
        draw.text(text_location, text, fill="white", font=font)

    return image
