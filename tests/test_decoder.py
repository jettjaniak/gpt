import torch
import pytest
import itertools

from gpt.decoder import Decoder
from tests import utils


@pytest.mark.parametrize(
    "n_heads, n_ctx, d_model, d_head",
    itertools.product([1, 3], repeat=4),
)
def test_decoder_batch_eq(
    n_heads: int,
    n_ctx: int,
    d_model: int,
    d_head: int,
):
    """Does the same input data in batch dimension result in the same output?"""
    decoder = Decoder(n_heads=n_heads, n_ctx=n_ctx, d_model=d_model, d_head=d_head, dropout_p=0.1)
    decoder.eval()
    embed, mask = utils.random_embed_mask_equal_batch(n_ctx, d_model)
    decoder_out_no_mask = decoder(embed)
    assert torch.allclose(decoder_out_no_mask[0], decoder_out_no_mask[1])
    decoder_out_with_mask = decoder(embed, mask)
    assert torch.allclose(decoder_out_with_mask[0], decoder_out_with_mask[1])


@pytest.mark.parametrize("n_heads, n_ctx, d_model, d_head, batch_size", itertools.product(range(1, 4), repeat=5))
def test_decoder_backward(n_heads, n_ctx, d_model, d_head, batch_size):
    decoder = Decoder(n_heads=n_heads, n_ctx=n_ctx, d_model=d_model, d_head=d_head, dropout_p=0.1)
    decoder.train()
    embed = torch.rand(batch_size, n_ctx, d_model)
    mask = utils.random_mask(batch_size, n_ctx)
    target = torch.rand(batch_size, n_ctx, d_model)

    output_with_mask = decoder(embed, mask)
    loss_with_mask = torch.mean((output_with_mask - target) ** 2)
    loss_with_mask.backward()

    output_no_mask = decoder(embed, None)
    loss_no_mask = torch.mean((output_no_mask - target) ** 2)
    loss_no_mask.backward()


def test_decoder_types_checked():
    n_ctx = 3
    d_model = 6
    decoder = Decoder(n_heads=2, n_ctx=n_ctx, d_model=d_model, d_head=3, dropout_p=0.1)
    with pytest.raises(TypeError):
        decoder(torch.rand(n_ctx, d_model))  # no batch dim
