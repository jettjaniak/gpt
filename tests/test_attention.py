import torch
import pytest
import itertools

from gpt.attention import Attention
from tests import utils

# TODO: multi-head tests
# TODO: unit tests


@pytest.mark.parametrize("n_ctx, d_model, d_head", itertools.product(range(1, 4), repeat=3))
def test_attention_batch_eq(n_ctx, d_model, d_head):
    """Does the same input data in batch dimension result in the same output?"""
    attn = Attention(d_model=d_model, d_head=d_head, dropout_p=0.1)
    attn.eval()
    embed, mask = utils.random_embed_mask_equal_batch(n_ctx, d_model)
    attn_out_with_mask = attn(embed, mask)
    assert torch.allclose(attn_out_with_mask[0], attn_out_with_mask[1])
    attn_out_no_mask = attn(embed, None)
    assert torch.allclose(attn_out_no_mask[0], attn_out_no_mask[1])


@pytest.mark.parametrize("n_ctx, d_model, d_head, batch_size", itertools.product(range(1, 4), repeat=4))
def test_attention_backward(n_ctx, d_model, d_head, batch_size):
    attn = Attention(d_model=d_model, d_head=d_head, dropout_p=0.1)
    attn.train()
    embed = torch.rand(batch_size, n_ctx, d_model)
    mask = utils.random_mask(batch_size, n_ctx)
    target = torch.rand(batch_size, n_ctx, d_head)

    output_with_mask = attn(embed, mask)
    loss_with_mask = torch.mean((output_with_mask - target) ** 2)
    loss_with_mask.backward()

    output_no_mask = attn(embed, None)
    loss_no_mask = torch.mean((output_no_mask - target) ** 2)
    loss_no_mask.backward()


def test_attention_types_checked():
    attn = Attention(d_model=2, d_head=2, dropout_p=0.1)
    with pytest.raises(TypeError):
        attn(torch.rand(2, 2), mask=None)  # no batch dim
