import torch
import pytest
import itertools

from gpt.decoder import Decoder


@pytest.mark.parametrize(
    "n_mlp_layers, n_heads, seq_len, d_embed, d_key, d_batch",
    itertools.product([1, 3], repeat=6),
)
def test_decoder(
    n_mlp_layers: int,
    n_heads: int,
    seq_len: int,
    d_embed: int,
    d_key: int,
    d_batch: int,
):
    decoder = Decoder(
        n_mlp_layers=n_mlp_layers,
        n_heads=n_heads,
        seq_len=seq_len,
        d_embed=d_embed,
        d_key=d_key,
    )
    embed = torch.rand(d_batch, seq_len, d_embed)
    decoder(embed)
