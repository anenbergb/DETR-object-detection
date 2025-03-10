import torch
from torch import nn


def positional_encoding(x, y, num_pos_feats=128, temperature=10000):
    """
    Generates sine-cosine positional embeddings.

    It is important that the x-coordinates, and y-coordinates are normalized to [0, 1]
    so that the model can learn relative positional relationships, regardless of the
    input image's dimensions.

    Args:
        x: Normalized x-coordinates (tensor).
        y: Normalized y-coordinates (tensor).
        num_pos_feats: Number of positional features.
        temperature: Temperature parameter for frequency scaling.

    Returns:
        Positional embeddings (tensor).
    """
    scale = 2 * torch.pi
    x = x * scale
    y = y * scale

    # generate a set of exponentially spaced frequencies for the sine and cosine
    # positional embeddings. These frequencies control the wavelengths of the sine
    # and cosine waves, allowing the model to capture positional information
    # at different scales.
    # The frequencies step by 2 to ensure that the frequencies used for
    # sine and cosines that are paired together are the same.
    # e.g. [0,0,2,2,4,4,6,6,...,62,62]
    dim_t = torch.arange(0, num_pos_feats, 2, dtype=torch.float32, device=x.device)
    # dim_t / num_pos_feats normalizes the range to [0,1]
    # {temperature**0 = 1, ...., temperature**1} exponential spacing of wavelengths
    # from lower frequencies (longer wavelength) to higher frequencies
    # (shorter wavelength)
    dim_t = temperature ** (dim_t / num_pos_feats)

    # Dividing the normalized coordinates by the frequency is equivalent to multipying by the period.
    # We're scaling the coordinates based on the wavelength of the sine and cosine waves.
    # We're converting the normalized coordinates [0, 2pi] into a phase value for the sine and cosine functions,
    # which is what the sine and consine functions expect as their input.
    # x[:,:,None] / dim_t broadcasting that each element in x is divided by each element of dim_t
    # to create a new tensor of shape (batch_size, height, width, num_pos_feats)
    # Each element pos_x[b, h, w, i] represents the normalized x-coordinate at position (h, w) in batch b,
    # divided by the i-th frequency in dim_t.
    pos_x = x[..., None] / dim_t
    pos_y = y[..., None] / dim_t
    # stack then flatten() will interleave the sine and cosine
    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)
    return pos


def batch_positional_encoding(batch_shape, heights, widths, num_pos_feats=128, temperature=10000, device="cpu"):
    batch_grid_x = torch.zeros(batch_shape, dtype=torch.float32, device=device)
    batch_grid_y = torch.zeros(batch_shape, dtype=torch.float32, device=device)
    for batch_i, (height, width) in enumerate(zip(heights, widths)):
        x_axis = torch.linspace(0, 1, width)
        y_axis = torch.linspace(0, 1, height)
        grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
        batch_grid_x[batch_i, :height, :width] = grid_x
        batch_grid_y[batch_i, :height, :width] = grid_y

    return positional_encoding(batch_grid_x, batch_grid_y, num_pos_feats, temperature)


class PositionalEncoding(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(
        self,
        embed_height: int,
        embed_width: int,
        image_heights: torch.Tensor,
        image_widths: torch.Tensor,
        scaling_factor: int = 32,
    ):
        """
        Downscale the image heights and widths by the scaling factor to match the embed height and widths
        """
        batch_shape = (len(image_heights), embed_height, embed_width)
        scaled_heights = torch.ceil(image_heights / scaling_factor).to(torch.int)
        scaled_widths = torch.ceil(image_widths / scaling_factor).to(torch.int)
        return batch_positional_encoding(
            batch_shape,
            scaled_heights,
            scaled_widths,
            self.num_pos_feats,
            self.temperature,
            device=image_heights.device,
        )
