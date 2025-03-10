from dataclasses import dataclass, field
import torch
from torch import nn
import math
from typing import Optional
from torchvision.models import get_model
from torchvision.ops import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter

from detr.position_encoding import PositionalEncoding


@dataclass
class DETRConfig:
    backbone: str = "resnet50"
    # positional encoding
    temperature: int = 10000

    num_queries: int = 100
    num_enc_layers: int = 6
    num_dec_layers: int = 6
    num_attention_heads: int = 8
    hidden_size: int = 256
    ffn_scale_factor: int = 8  # 256x8 = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5

    num_classes: int = 80


class DETR(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.config = config
        self.backbone = Backbone(config)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, config.hidden_size, kernel_size=1)
        self.position_embedding = PositionalEncoding(
            num_pos_feats=config.hidden_size // 2, temperature=config.temperature
        )
        self.query_embeding = nn.Embedding(config.num_queries, config.hidden_size)  # (100, 256)

        self.encoder = Encoder(config)
        # self.decoder = Decoder(config)

        self.class_embedding = nn.Linear(config.hidden_size, config.num_classes + 1)
        self.bbox_embedding = MLP(config.hidden_size, config.hidden_size, output_dim=4, num_layers=3)

    def forward(self, images: torch.Tensor, heights: torch.Tensor, widths: torch.Tensor):
        x = self.backbone(images)
        x = self.input_proj(x)
        B, C, H, W = x.shape
        pos_embd = self.position_embedding(H, W, heights, widths, self.backbone.scale)  # (B,256,feat_height,feat_width)
        attn_mask = self.make_attention_mask(H, W, heights, widths, self.backbone.scale)  # (B, feat_height, feat_width)
        # Flatten the spatial dimensions
        # (B, 256, H, W) -> (B, 256, H*W) -> (B, H*W, 256)
        x = x.flatten(2).permute(0, 2, 1)
        pos_embd = pos_embd.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)
        key_padding_mask = attn_mask.flatten(1)  # (B, H*W)
        # query_embed = self.query_embeding.weight.unsqueeze(1).repeat(1, B, 1)  # (100, B, 256)
        query_embed = self.query_embeding.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, 100, 256)

        encoded_memory = self.encoder(x, position_embedding=pos_embd)
        return encoded_memory

    def make_attention_mask(
        self,
        embed_height: int,
        embed_width: int,
        image_heights: torch.Tensor,
        image_widths: torch.Tensor,
        scaling_factor: int = 32,
    ):
        """
        Create a mask to prevent attention to padding tokens.
        """
        mask = torch.zeros(
            (len(image_heights), embed_height, embed_width), dtype=torch.int, device=image_heights.device
        )
        scaled_heights = torch.ceil(image_heights / scaling_factor).to(torch.int)
        scaled_widths = torch.ceil(image_widths / scaling_factor).to(torch.int)
        for i, (height, width) in enumerate(zip(scaled_heights, scaled_widths)):
            mask[i, :height, :width] = 1
        return mask


class Encoder(nn.Module):
    """pre-LN Transformer Encoder"""

    def __init__(self, config: DETRConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_enc_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, position_embedding: Optional[torch.Tensor] = None):

        for layer in self.layer:
            x = layer(x, position_embedding=position_embedding)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.self_attention = ScaledDotProductAttention(config)
        self.ffn = FFN(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, position_embedding: Optional[torch.Tensor] = None):
        x_attn = self.norm1(x)
        query = key = x_attn + position_embedding
        x = x + self.self_attention(query, key, value=x_attn)
        x = x + self.ffn(self.norm2(x))
        return x


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    Args:
        config: DETRConfig
    """

    def __init__(self, config: DETRConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        # heads are parallel streams, and outputs get concatenated.
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # output projection
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout_attn = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hidden_size = config.hidden_size  # 256
        self.n_head = config.num_attention_heads  # 8
        self.head_size = config.hidden_size // config.num_attention_heads  # 32

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the attention mechanism.

        Args:
            query: torch.Tensor
                Query tensor.
            key: torch.Tensor
                Key tensor.
            value: torch.Tensor
                Value tensor.
            attention_mask: torch.Tensor
                If specified, a mask of shape (N,L) preventing attention to certain positions

            key_padding_mask: torch.Tensor
                If specified, a mask of shape (N,L) indicating which elements within key to ignore
                for the purpose of attention (i.e. treat as “padding”).
                Binary and float masks are supported.
                For a binary mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention.
                For a float mask, it will be directly added to the corresponding key value.


        Returns:
            torch.Tensor: Output tensor after applying attention.
        """

        # TODO: make sure this works for cross attention, where I think the
        # dimensions of query, key, and value are different
        B, T, C = query.size()  # batch size, sequence length, embedding dimensionality

        # calculate query, key, value for all heads in batch
        # C is hidden_size, which is 256 in DETR
        # nh is "number of heads", which is 8 in DETR
        # hs is "head size", which is C // nh = 256 // 8 = 32

        query = self.query_proj(query)  # (B, T, C)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # (B, T, C) -> (B, T, nh, C/nh) = (B, T, nh, 32) --transpose(1,2)--> (B, nh, T, 32)
        q = query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 32)
        k = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 32)
        v = value.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 32)

        # attention multiplies the head_size dimension (T,32) x (32,T) = (T,T)
        # (B, nh, T, 32) x (B, nh, 32, T) -> (B, nh, T, T)
        att = q @ k.transpose(2, 3)
        att = att / math.sqrt(self.head_size)

        # attention mask is a binary mask of shape (B,T) that is 1 for positions we want to attend to
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        # Broadcast to (B, nh, T, T) by applying it to the key dimension
        # Mask out padding by setting scores to -inf where attn_mask is 0
        # att = att.masked_fill(attention_mask == 0, torch.finfo(att.dtype).min)  # (B, nh, T, T)

        # att describes the relation between the tokens in the sequence
        # how much token 0 should be a mixture of tokens 0 through T
        att = nn.functional.softmax(att, dim=-1)
        # Randomly sets some attention weights to zero during training,
        # meaning certain key-value pairs are ignored for that forward pass.
        # This prevents the model from over-relying on specific attention patterns.
        att = self.dropout_attn(att)

        # re-mix the value tokens, by multiplying each token by the corresponding
        # weights in the attention matrix. Do this across all 64 dimensions
        y = att @ v  # (B, nh, T, T) x (B, nh, T, 32) -> (B, nh, T, 32)

        # (B, nh, T, 32) -> (B, T, nh, 32) -> (B, T, nh*32 = 8*32 = 256)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.output_proj(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):
    """Very simple multi-layer perceptron"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FFN(nn.Module):
    """
    Feed-Forward Network (FFN) used in DETR.

    Args:
        config: DETRConfig
            Configuration for the BERT model.
    """

    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.ffn_scale_factor),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size * config.ffn_scale_factor, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

    def forward(self, x):
        """
        Forward pass for the feed-forward network.

        Args:
            x: torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the feed-forward network.
        """
        return self.layers(x)


class Backbone(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.config = config
        assert config.backbone in ("resnet50", "resnet101"), "Only resnet50 and resnet101 backbones are supported"
        model = get_model(config.backbone, weights="DEFAULT", norm_layer=FrozenBatchNorm2d)
        self.backbone = IntermediateLayerGetter(model, return_layers={"layer4": "final_feature_map"})
        self.num_channels = 2048
        self.scale = 32

    def forward(self, x: torch.Tensor):
        return self.backbone(x)["final_feature_map"]
