import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Tuple, Optional

from gpt.linear_normal import LinearNormal

patch_typeguard()

HeadTensor = TensorType["batch", "ctx", "head", torch.float32]
ModelTensor = TensorType["batch", "ctx", "model", torch.float32]
MaskTensor = TensorType["batch", "ctx", "ctx", torch.bool]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_head: int):
        super().__init__()
        self.heads = nn.ModuleList([Attention(d_model, d_head) for _ in range(n_heads)])
        self.w_o = LinearNormal(n_heads * d_head, d_model, bias=False)

    @typechecked
    def forward(self, embed: ModelTensor, mask: Optional[MaskTensor]) -> ModelTensor:
        # it's sequential, we could parallelize it by adding a dimension to Attention instead
        # [(batch, ctx, head), ...] -> (batch, ctx, n_heads * head)
        heads_out = torch.cat([head(embed, mask) for head in self.heads], dim=2)
        # -> (batch, ctx, model)
        return self.w_o(heads_out)


# TODO: masking?
class Attention(nn.Module):
    """Single attention head"""

    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        # Weights for Key, Query and Value
        self.w_k = LinearNormal(d_model, d_head, bias=False)
        self.w_q = LinearNormal(d_model, d_head, bias=False)
        self.w_v = LinearNormal(d_model, d_head, bias=False)
        self.norm = d_head**0.5

    @typechecked
    def compute_kqv(self, embed: ModelTensor) -> Tuple[HeadTensor, HeadTensor, HeadTensor]:
        return (
            self.w_k(embed),
            self.w_q(embed),
            self.w_v(embed),
        )

    @typechecked
    def compute_prob(
        self, q: HeadTensor, k: HeadTensor, mask: Optional[MaskTensor]
    ) -> TensorType["batch", "ctx", "ctx"]:
        # (batch, ctx, head) -> (batch, head, ctx)
        k_t = torch.transpose(k, dim0=1, dim1=2)
        # (batch, ctx, head), (batch, head, ctx) -> (batch, ctx, ctx)
        logits = (q @ k_t) / self.norm
        if mask is not None:
            logits[~mask] = -float("inf")
        # keeps the (batch, ctx, ctx) size, sum along last dim is 1
        return torch.softmax(logits, dim=2)

    @typechecked
    def forward(self, embed: ModelTensor, mask: Optional[MaskTensor]) -> HeadTensor:
        k, q, v = self.compute_kqv(embed)
        prob = self.compute_prob(q, k, mask)
        # (batch, ctx, ctx) @ (batch, ctx, head) -> (batch, ctx, head)
        # each row in prob is a distribution over seq tokens, and each column of v corresponds to one token
        # we are weighting and summing values
        return prob @ v
