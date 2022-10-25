import torch
from torch import nn
from torchtyping import TensorType
from typing import Optional

from gpt.linear_normal import LinearNormal
from gpt.types import ModelTensor, PadMaskTensor, FullMaskTensor, HeadTensor


class MultiHeadAttention(nn.Module):
    """Concatenate multiple attention heads outputs and apply linear transformation"""

    def __init__(self, n_heads: int, d_model: int, d_head: int, dropout_p: float):
        super().__init__()
        self.heads = nn.ModuleList([Attention(d_model, d_head, dropout_p) for _ in range(n_heads)])
        # TODO: should it really be bias=False?
        self.w_o = LinearNormal(n_heads * d_head, d_model, bias=False)

    def forward(self, embed: ModelTensor, pad_mask: PadMaskTensor) -> ModelTensor:
        # [(batch, ctx, head), ...] -> (batch, ctx, n_heads * head)
        heads_out = torch.cat([head(embed, pad_mask) for head in self.heads], dim=2)
        # -> (batch, ctx, model)
        return self.w_o(heads_out)


class Attention(nn.Module):
    """Single attention head"""

    def __init__(self, d_model: int, d_head: int, dropout_p: float):
        super().__init__()
        # Weights for Key, Query and Value
        # TODO: should it really be bias=False?
        self.w_k = LinearNormal(d_model, d_head, bias=False)
        self.w_q = LinearNormal(d_model, d_head, bias=False)
        self.w_v = LinearNormal(d_model, d_head, bias=False)
        self.norm = d_head**0.5
        self.softmax_dropout = nn.Dropout(p=dropout_p)

    def forward(self, embed: ModelTensor, pad_mask: PadMaskTensor) -> HeadTensor:
        k = self.w_k(embed)
        q = self.w_q(embed)
        v = self.w_v(embed)
        full_mask = make_full_mask(pad_mask)
        softmax = self.softmax_dropout(compute_attention_softmax(q, k, self.norm, full_mask))
        # (batch, ctx, ctx) @ (batch, ctx, head) -> (batch, ctx, head)
        # each row in prob is a distribution over seq tokens, and each column of v corresponds to one token
        # we are weighting and summing values
        return softmax @ v


def compute_attention_softmax(
    q: HeadTensor, k: HeadTensor, norm: float, full_mask: FullMaskTensor
) -> TensorType["batch", "ctx", "ctx"]:
    # (batch, ctx, head) -> (batch, head, ctx)
    k_t = torch.transpose(k, dim0=1, dim1=2)
    # (batch, ctx, head), (batch, head, ctx) -> (batch, ctx, ctx)
    logits = (q @ k_t) / norm
    logits[~full_mask] = -float("inf")
    # keeps the (batch, ctx, ctx) size, sum along last dim is 1
    return torch.softmax(logits, dim=2)


def make_full_mask(pad_mask: PadMaskTensor) -> FullMaskTensor:
    batch_size, n_ctx = pad_mask.size()
    # 0-th element can attend to itself
    peeking_mask = torch.tril(torch.ones(1, n_ctx, n_ctx, dtype=torch.bool)).expand(batch_size, -1, -1)
    # (batch, ctx) -> (batch, ctx, ctx)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, n_ctx, -1)  # mask the same tokens each time
    return pad_mask & peeking_mask
