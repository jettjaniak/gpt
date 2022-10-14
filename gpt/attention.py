import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Tuple

patch_typeguard()

KeyTensor = TensorType["batch", "seq", "key"]
EmbedTensor = TensorType["batch", "seq", "embed"]

# TODO: use squeeze


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_key: int):
        super().__init__()
        self.heads = nn.ModuleList([Attention(d_embed, d_key) for _ in range(n_heads)])
        self.w_o = nn.Parameter(torch.normal(mean=0, std=0.02, size=(d_embed, n_heads * d_key)))

    @typechecked
    def forward(self, embed: EmbedTensor) -> EmbedTensor:
        # it's sequential, we could parallelize it by adding "head" dimension to Attention instead
        # [(batch, seq, key), ...] -> (batch, seq, heads * key)
        heads_out = torch.cat([head(embed) for head in self.heads], dim=2)
        # -> (batch, seq, heads * key, 1)
        heads_out_ = heads_out.view(*heads_out.size(), 1)
        # (embed, heads * key) @  (batch, seq, heads * key, 1) -> (batch, seq, embed, 1)
        ret_4dim = self.w_o @ heads_out_
        # -> (batch, seq, embed)
        return ret_4dim.view(*ret_4dim.size()[:-1])


# TODO: masking?
class Attention(nn.Module):
    def __init__(self, d_embed: int, d_key: int):
        super().__init__()
        # Weights for Key, Query and Value
        self.w_k = nn.Parameter(torch.normal(mean=0, std=0.02, size=(d_key, d_embed)))
        self.w_q = nn.Parameter(torch.normal(mean=0, std=0.02, size=(d_key, d_embed)))
        self.w_v = nn.Parameter(torch.normal(mean=0, std=0.02, size=(d_key, d_embed)))
        self.norm = d_key**0.5

    @staticmethod
    @typechecked
    def _compute_k_or_q_or_v(
        weights: TensorType["key", "embed"],
        embed_: TensorType["batch", "seq", "embed", 1],
    ) -> KeyTensor:
        # (key, embed) @ (batch, seq, embed, 1) -> (batch, seq, key, 1)
        ret_4dim = weights @ embed_
        # -> (batch, seq, key)
        return ret_4dim.view(ret_4dim.size()[:-1])

    @typechecked
    def compute_kqv(self, embed: EmbedTensor) -> Tuple[KeyTensor, KeyTensor, KeyTensor]:
        embed_ = embed.view(*embed.size(), 1)
        return (
            self._compute_k_or_q_or_v(self.w_k, embed_),
            self._compute_k_or_q_or_v(self.w_q, embed_),
            self._compute_k_or_q_or_v(self.w_v, embed_),
        )

    @typechecked
    def compute_prob(self, q: KeyTensor, k: KeyTensor) -> TensorType["batch", "seq", "seq"]:
        # (batch, seq, key) -> (batch, key, seq)
        k_t = torch.transpose(k, dim0=1, dim1=2)
        # (batch, seq, key), (batch, key, seq) -> (batch, seq, seq)
        logits = (q @ k_t) / self.norm
        # keeps the (batch, seq, seq) size, sum along last dim is 1
        return torch.softmax(logits, dim=2)

    @typechecked
    def forward(self, embed: EmbedTensor) -> KeyTensor:
        k, q, v = self.compute_kqv(embed)
        prob = self.compute_prob(q, k)
        # (batch, seq, seq) @ (batch, seq, key) -> (batch, seq, key)
        # each row in prob is a distribution over seq tokens, and each column of v corresponds to one token
        # we are weighting and summing values them
        return prob @ v
