import torch
from torch import nn
from torchtyping import TensorType
from typing import Optional

from gpt.linear_normal import LinearNormal
from gpt.types import ModelTensor, MaskTensor, HeadTensor


class MultiHeadAttention(nn.Module):
    """Concatenate multiple attention heads outputs and apply linear transformation"""

    def __init__(self, n_heads: int, d_model: int, d_head: int, dropout_p: float):
        super().__init__()
        self.heads = nn.ModuleList([Attention(d_model, d_head, dropout_p) for _ in range(n_heads)])
        self.w_o = LinearNormal(n_heads * d_head, d_model, bias=False)

    def forward(self, embed: ModelTensor, mask: Optional[MaskTensor]) -> ModelTensor:
        # it's sequential, we could parallelize it by adding a dimension to Attention instead
        # [(batch, ctx, head), ...] -> (batch, ctx, n_heads * head)
        heads_out = torch.cat([head(embed, mask) for head in self.heads], dim=2)
        # -> (batch, ctx, model)
        return self.w_o(heads_out)


class Attention(nn.Module):
    """Single attention head"""

    def __init__(self, d_model: int, d_head: int, dropout_p: float):
        super().__init__()
        # Weights for Key, Query and Value
        self.w_k = LinearNormal(d_model, d_head, bias=False)
        self.w_q = LinearNormal(d_model, d_head, bias=False)
        self.w_v = LinearNormal(d_model, d_head, bias=False)
        self.norm = d_head**0.5
        self.softmax_dropout = nn.Dropout(p=dropout_p)

    def forward(self, embed: ModelTensor, mask: Optional[MaskTensor]) -> HeadTensor:
        k = self.w_k(embed)
        q = self.w_q(embed)
        v = self.w_v(embed)
        softmax = self.softmax_dropout(compute_attention_softmax(q, k, self.norm, mask))
        # (batch, ctx, ctx) @ (batch, ctx, head) -> (batch, ctx, head)
        # each row in prob is a distribution over seq tokens, and each column of v corresponds to one token
        # we are weighting and summing values
        return softmax @ v


def compute_attention_softmax(
    q: HeadTensor, k: HeadTensor, norm: float, mask: Optional[MaskTensor]
) -> TensorType["batch", "ctx", "ctx"]:
    # (batch, ctx, head) -> (batch, head, ctx)
    k_t = torch.transpose(k, dim0=1, dim1=2)
    # (batch, ctx, head), (batch, head, ctx) -> (batch, ctx, ctx)
    logits = (q @ k_t) / norm
    if mask is not None:
        logits[~mask] = -float("inf")
    # keeps the (batch, ctx, ctx) size, sum along last dim is 1
    return torch.softmax(logits, dim=2)
