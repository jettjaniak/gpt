import torch
from torch import nn
from torchtyping import patch_typeguard
from typeguard import typechecked
from typing import Optional

from gpt.attention import MultiHeadAttention, ModelTensor, MaskTensor
from gpt.linear_normal import LinearNormal

patch_typeguard()


class Decoder(nn.Module):
    def __init__(self, n_heads: int, n_ctx: int, d_model: int, d_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_heads, d_model, d_head)
        d_ff = 4 * d_model
        self.ff = nn.Sequential(
            LinearNormal(d_model, d_ff, bias=True), nn.GELU(), LinearNormal(d_ff, d_model, bias=True)
        )
        self.layer_norm_attn = nn.LayerNorm([n_ctx, d_model])
        self.layer_norm_mlp = nn.LayerNorm([n_ctx, d_model])

    @typechecked
    def forward(self, embed: ModelTensor, mask: Optional[MaskTensor] = None) -> ModelTensor:
        attn_out = self.attn(embed, mask)
        add_norm_attn_out = self.layer_norm_attn(attn_out + embed)
        mlp_out = self.ff(add_norm_attn_out)
        return self.layer_norm_mlp(mlp_out + add_norm_attn_out)
