from dataclasses import dataclass, field
import torch
from torch import nn
import math
from typing import Optional


@dataclass
class DETRConfig:
    backbone: str = "resnet50"
    position_embedding_type: str = "sine"

    num_queries: int = 100
    num_enc_layers: int = 6
    num_dec_layers: int = 6
    num_attention_heads: int = 8
    hidden_size: int = 256
    feedforward_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5




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

    def forward(self, x: torch.Tensor, attention_mask: torch.LongTensor) -> torch.Tensor:
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
        # C is hidden_size, which is 256 in BERT
        # nh is "number of heads", which is 12 in BERT
        # hs is "head size", which is C // nh = 768 // 12 = 64 in BERT

        query = self.query_proj(x)  # (B, T, C)
        

        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.hidden_size, dim=2)  # (B, T, C) x 3
        # (B, T, C) -> (B, T, nh, C/nh) = (B, T, nh, 64) --transpose(1,2)--> (B, nh, T, 64)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 64)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 64)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, 64)

        # attention multiplies the head_size dimension (T,64) x (64,T) = (T,T)
        # (B, nh, T, 64) x (B, nh, 64, T) -> (B, nh, T, T)
        att = q @ k.transpose(2, 3)
        att = att / math.sqrt(self.head_size)

        # attention mask is a binary mask of shape (B,T) that is 1 for positions we want to attend to
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        # Broadcast to (B, nh, T, T) by applying it to the key dimension
        # Mask out padding by setting scores to -inf where attn_mask is 0
        att = att.masked_fill(attention_mask == 0, torch.finfo(att.dtype).min)  # (B, nh, T, T)

        # att describes the relation between the tokens in the sequence
        # how much token 0 should be a mixture of tokens 0 through T
        att = nn.functional.softmax(att, dim=-1)
        # Randomly sets some attention weights to zero during training,
        # meaning certain key-value pairs are ignored for that forward pass.
        # This prevents the model from over-relying on specific attention patterns.
        att = self.dropout_attn(att)

        # re-mix the value tokens, by multiplying each token by the corresponding
        # weights in the attention matrix. Do this across all 64 dimensions
        y = att @ v  # (B, nh, T, T) x (B, nh, T, 64) -> (B, nh, T, 64)
        # The masked values (0 values in the attention mask), e.g. the values
        # from t:T in the sequence of length T, will have random noisy values
        # in the (:,:,t:T,:) region of the tensor.
        # Obvously these values should be ignored in the final output.

        # (B, nh, T, 64) -> (B, T, nh, 64) -> (B, T, nh*64 = 12*64 = 768)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        y = self.dropout(y)
        return y
