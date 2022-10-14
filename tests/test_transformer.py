import torch
import pytest
import itertools

from gpt.transformer import Transformer
from tests import utils


@pytest.mark.parametrize(
    "vocab_size, n_layers, n_heads, n_ctx, d_model, d_head",
    itertools.product([1, 3], repeat=6),
)
def test_transformer_batch_eq(
    vocab_size: int,
    n_layers: int,
    n_heads: int,
    n_ctx: int,
    d_model: int,
    d_head: int,
):
    """Does the same input data in batch dimension result in the same output?"""
    transformer = Transformer(
        vocab_size=vocab_size, n_layers=n_layers, n_heads=n_heads, n_ctx=n_ctx, d_model=d_model, d_head=d_head
    )
    embed, mask = utils.random_embed_mask_equal_batch(n_ctx, d_model)
    transformer_out = transformer(embed, mask)
    assert torch.allclose(transformer_out[0], transformer_out[1])


@pytest.mark.parametrize(
    "vocab_size, n_layers, n_heads, n_ctx, d_model, d_head, batch_size", itertools.product([1, 5], repeat=7)
)
def test_transformer_backward(
    vocab_size: int, n_layers: int, n_heads: int, n_ctx: int, d_model: int, d_head: int, batch_size: int
):
    transformer = Transformer(
        vocab_size=vocab_size, n_layers=n_layers, n_heads=n_heads, n_ctx=n_ctx, d_model=d_model, d_head=d_head
    )
    embed = torch.rand(batch_size, n_ctx, d_model)
    mask = utils.random_mask(batch_size, n_ctx)
    target = torch.rand(batch_size, n_ctx, d_model)

    output = transformer(embed, mask)
    loss = torch.mean((output - target) ** 2)
    loss.backward()
