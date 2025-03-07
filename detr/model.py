from dataclasses import dataclass, field
import torch
from torch import nn
import math
from typing import Optional
from torchvision.models._utils import IntermediateLayerGetter

@dataclass
class DETRConfig:
    backbone: str = "resnet50"
    position_embedding_type: str = "sine"

    num_queries: int = 100
    num_enc_layers: int = 6
    num_dec_layers: int = 6
    num_attention_heads: int = 8
    hidden_size: int = 256
    ffn_scale_factor: int = 8 # 256x8 = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5

    num_classes: int = 80


class DETR(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.config = config

        self.class_embed = nn.Linear(self.hidden_size, self.num_classes + 1)
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_size)
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # create backbone
        # self.backbone = Backbone(config)

        # create position embeddings
        self.position_embedding = PositionEmbedding(config)

        # create class and box prediction heads
        self.class_embed = nn.Linear(config.hidden_size, config.num_queries + 1)
        self.bbox_embed = nn.Linear(config.hidden_size, 4)


class Encoder(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_enc_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor
        position_embedding: Optional[Tensor] = None):

        for layer in self.layer:
            x = layer(x, position_embedding = position_embedding)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.self_attention = ScaledDotProductAttention(config)
        self.ffn = FFN(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, position_embedding: Optional[Tensor] = None):
        x_attn = self.norm1(x)
        query = key = x_attn + position_embedding
        x = x + self.self_attention(query, key, value=x_attn)
        x = x + self.ffn(self.norm2(x))
        return x

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
            nn.Linear(config.hidden_size, config.hidden_size * self.ffn_scale_factor),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size * self.ffn_scale_factor, config.hidden_size),
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
        # attention_mask: torch.LongTensor
        # key_padding_mask: Optional[Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the attention mechanism.

        Args:
            x: torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # calculate query, key, value for all heads in batch
        # C is hidden_size, which is 256 in DETR
        # nh is "number of heads", which is 8 in DETR
        # hs is "head size", which is C // nh = 256 // 8 = 32

        query = self.query_proj(x)  # (B, T, C)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # (B, T, C) -> (B, T, nh, C/nh) = (B, T, nh, 32) --transpose(1,2)--> (B, nh, T, 32)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 32)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 32)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 32)

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
        y = self.c_proj(y)
        y = self.dropout(y)
        return y
