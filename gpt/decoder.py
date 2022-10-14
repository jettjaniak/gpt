import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Tuple

from gpt.attention import MultiHeadAttention, EmbedTensor
from gpt.mlp import MLP

patch_typeguard()


class Decoder(nn.Module):
    def __init__(self, n_mlp_layers: int, n_heads: int, seq_len: int, d_embed: int, d_key: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_heads, d_embed, d_key)
        self.mlp = MLP(n_mlp_layers, layer_width=d_embed)
        self.layer_norm_attn = nn.LayerNorm([seq_len, d_embed])
        self.layer_norm_mlp = nn.LayerNorm([seq_len, d_embed])

    @typechecked
    def forward(self, embed: EmbedTensor) -> EmbedTensor:
        attn_out = self.attn(embed)
        add_norm_attn_out = self.layer_norm_attn(attn_out + embed)
        mlp_out = self.mlp(add_norm_attn_out)
        return self.layer_norm_mlp(mlp_out + add_norm_attn_out)
