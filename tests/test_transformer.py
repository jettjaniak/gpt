import torch
import pytest
import itertools

from gpt.transformer import Transformer


@pytest.mark.parametrize(
    "n_tokens, n_decoders, n_mlp_layers, n_heads, seq_len, d_embed, d_key, d_batch",
    itertools.product([1, 3], repeat=8),
)
def test_transformer(
    n_tokens: int,
    n_decoders: int,
    n_mlp_layers: int,
    n_heads: int,
    seq_len: int,
    d_embed: int,
    d_key: int,
    d_batch: int,
):
    transformer = Transformer(
        n_tokens=n_tokens,
        n_decoders=n_decoders,
        n_decoder_mlp_layers=n_mlp_layers,
        n_attn_heads=n_heads,
        seq_len=seq_len,
        d_embed=d_embed,
        d_key=d_key,
    )
    embed = torch.rand(d_batch, seq_len, d_embed)
    transformer(embed)
