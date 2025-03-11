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

    num_object_queries: int = 100
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
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
        self.object_query_embedding = nn.Embedding(config.num_object_queries, config.hidden_size)  # (100, 256)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.class_embedding = nn.Linear(config.hidden_size, config.num_classes + 1)
        self.bbox_embedding = MLP(config.hidden_size, config.hidden_size, output_dim=4, num_layers=3)

    def forward(self, images: torch.Tensor, heights: torch.Tensor, widths: torch.Tensor):
        x = self.backbone(images)
        x = self.input_proj(x)
        B, C, H, W = x.shape
        pos_embd = self.position_embedding(H, W, heights, widths, self.backbone.scale)  # (B,256,feat_height,feat_width)
        image_padding_mask = self.make_image_padding_mask(
            H, W, heights, widths, self.backbone.scale
        )  # (B, feat_height, feat_width)
        # Flatten the spatial dimensions
        # (B, 256, H, W) -> (B, 256, H*W) -> (B, H*W, 256)
        x = x.flatten(2).permute(0, 2, 1)
        pos_embd = pos_embd.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)
        image_padding_mask = image_padding_mask.flatten(1)  # (B, H*W)
        query_embed = self.object_query_embedding.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, 100, 256)

        # Encoded image tokens (B, H*W, 256)
        encoded_image_tokens = self.encoder(x, position_embedding=pos_embd, key_padding_mask=image_padding_mask)
        decoded_object_queries = self.decoder(
            encoded_image_tokens,
            position_embedding=pos_embd,
            object_query_embedding=query_embed,
            key_padding_mask=image_padding_mask,
        )
        return decoded_object_queries

    def make_image_padding_mask(
        self,
        embed_height: int,
        embed_width: int,
        image_heights: torch.Tensor,
        image_widths: torch.Tensor,
        scaling_factor: int = 32,
    ):
        """
        Returns binary mask of shape (B, embed_height, embed_width) containing True on padded pixels.
        """
        mask = torch.zeros(
            (len(image_heights), embed_height, embed_width), dtype=torch.bool, device=image_heights.device
        )
        scaled_heights = torch.ceil(image_heights / scaling_factor).to(torch.int)
        scaled_widths = torch.ceil(image_widths / scaling_factor).to(torch.int)
        for i, (height, width) in enumerate(zip(scaled_heights, scaled_widths)):
            mask[i, height:, width:] = True
        return mask


class Decoder(nn.Module):
    """pre-LN Transformer Decoder"""

    def __init__(self, config: DETRConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        encoded_image_tokens: torch.Tensor,
        position_embedding: torch.Tensor,
        object_query_embedding: torch.Tensor,
        key_padding_mask: torch.BoolTensor,
    ):
        x = torch.zeros_like(object_query_embedding)
        for layer in self.layers:
            x = layer(x, encoded_image_tokens, object_query_embedding, position_embedding, key_padding_mask)
        # Consider returning intermediates
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.self_attention = ScaledDotProductAttention(config)
        self.cross_attention = ScaledDotProductAttention(config)

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FFN(config)

    def forward(
        self,
        x: torch.Tensor,  # (B, 100, 256)
        encoded_image_tokens: torch.Tensor,  # (B, H*W, 256)
        object_query_embedding: torch.Tensor,  # (B, 100, 256)
        position_embedding: torch.Tensor,  # (B, H*W, 256)
        key_padding_mask: torch.BoolTensor,  # (B, H*W)
    ):
        x_attn = self.norm1(x)
        query = key = x_attn + object_query_embedding
        x = x + self.self_attention(query, key, value=x_attn)

        x_attn = self.norm2(x)
        query = x_attn + object_query_embedding
        key = encoded_image_tokens + position_embedding
        x = x + self.cross_attention(query, key, value=encoded_image_tokens, key_padding_mask=key_padding_mask)

        x = x + self.ffn(self.norm3(x))
        return x


class Encoder(nn.Module):
    """pre-LN Transformer Encoder"""

    def __init__(self, config: DETRConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, position_embedding: torch.Tensor, key_padding_mask: torch.BoolTensor):
        for layer in self.layers:
            x = layer(x, position_embedding, key_padding_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.self_attention = ScaledDotProductAttention(config)
        self.ffn = FFN(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, position_embedding: torch.Tensor, key_padding_mask: torch.BoolTensor):
        x_attn = self.norm1(x)
        query = key = x_attn + position_embedding
        x = x + self.self_attention(query, key, value=x_attn, key_padding_mask=key_padding_mask)
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
        key_padding_mask: Optional[torch.BoolTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the attention mechanism.

        Args:
            query: torch.Tensor (B,L,C)
                Query tensor.
                L is target sequence length, C is hidden size
            key: torch.Tensor (B,S,C)
                Key tensor.
                S is source sequence length, C is hidden size
            value: torch.Tensor (B,S,C)
                Value tensor.
                S is source sequence length, C is hidden size
            key_padding_mask: torch.Tensor (B,S)
                S is source sequence length
                If provided, specified padding elements in the key sequence will
                be ignored by the attention.
                This is an binary mask.
                When the value is True, the corresponding value on the attention layer will be filled with -inf.

            attention_mask: torch.Tensor (L,S)
                If specified, a mask of shape (L,S) preventing attention to certain positions
                It could be used to enforce causality (auto-regressive models like decoders)
                or to apply custom attention patterns.
                Unlike key_padding_mask, which is specific to padding in the key sequence,
                attn_mask operates on the relationship between query and key positions.

                This is a binary mask
                True indicates positions that should be masked (not attended to).
                False indicates positions that can be attended to.

                Example causal attention mask where S=3, L=3:
                attn_mask = tensor([[False,  True,  True],  # Pos 0 attends to Pos 0 only
                                    [False, False,  True],  # Pos 1 attends to Pos 0, 1
                                    [False, False, False]]) # Pos 2 attends to all

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """

        # TODO: make sure this works for cross attention, where I think the
        # dimensions of query, key, and value are different
        B, L, C = query.size()  # batch size, target sequence length, embedding dimensionality
        S = key.size(1)  # source sequence length

        # calculate query, key, value for all heads in batch
        # C is hidden_size, which is 256 in DETR
        # nh is "number of heads", which is 8 in DETR
        # hs is "head size", which is C // nh = 256 // 8 = 32

        query = self.query_proj(query)  # (B, L, C)
        key = self.key_proj(key)  # (B, S, C)
        value = self.value_proj(value)

        # (B, L, C) -> (B, L, nh, C/nh) = (B, L, nh, 32) --transpose(1,2)--> (B, nh, L, 32)
        q = query.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, L, 32)
        k = key.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, S, 32)
        v = value.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, S, 32)

        # attention multiplies the head_size dimension (L,32) x (32,S) = (L,S)
        # (B, nh, L, 32) x (B, nh, 32, S) -> (B, nh, L, S)
        att = q @ k.transpose(2, 3)
        att = att / math.sqrt(self.head_size)

        if key_padding_mask is not None:
            # Reshape and expand to match attn_weights dimensions
            att = att.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), torch.finfo(att.dtype).min  # [B, 1, 1, S]
            )

        if attention_mask is not None:
            # attention_mask is shape (L,S) so it can be broadcast to (B, nh, L, S)
            att = att.masked_fill(attention_mask, torch.finfo(att.dtype).min)

        # attention mask (L,S) describes the relation between the tokens in the query sequence [0, L-1]
        # and the tokens in the source(key,value) sequence [0, S-1]
        # The attention value att[i,j]=p reflects how query token i should consist of 'p' percent
        # of the information from source token j
        #  sum(att[i,:]) = 1
        att = nn.functional.softmax(att, dim=-1)
        # Randomly sets some attention weights to zero during training,
        # meaning certain key-value pairs are ignored for that forward pass.
        # This prevents the model from over-relying on specific attention patterns.
        att = self.dropout_attn(att)

        # re-mix the value tokens, by multiplying each token by the corresponding
        # weights in the attention matrix. Do this across all 32 dimensions
        y = att @ v  # (B, nh, L, S) x (B, nh, S, 32) -> (B, nh, L, 32)

        # (B, nh, L, 32) -> (B, L, nh, 32) -> (B, L, nh*32 = 8*32 = 256)
        y = y.transpose(1, 2).contiguous().view(B, L, C)
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
