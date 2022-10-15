import torch
import pytest
import itertools
import math

from gpt.attention import Attention, compute_attention_softmax
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


def test_compute_attention_softmax():
    q = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).unsqueeze(dim=0)
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).unsqueeze(dim=0)
    softmax = torch.tensor([[1 / (math.e + 1), math.e / (math.e + 1)], [math.e / (math.e + 1), 1 / (math.e + 1)]])
    assert torch.allclose(compute_attention_softmax(q, k, norm=1, mask=None), softmax)
