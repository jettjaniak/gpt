import torch
import pytest
import itertools

from gpt.attention import Attention

# TODO: multi-head tests


@pytest.mark.parametrize("d_seq, d_embed, d_key", itertools.product(range(1, 4), repeat=3))
def test_attention_batch_eq(d_seq, d_embed, d_key):
    attn = Attention(d_embed=d_embed, d_key=d_key)
    single_embed = torch.rand(1, d_seq, d_embed)
    embed = torch.cat([single_embed, single_embed], dim=0)
    attn_out = attn(embed)
    assert torch.allclose(attn_out[0], attn_out[1])


@pytest.mark.parametrize("d_embed, d_key", itertools.product(range(1, 4), repeat=2))
def test_attention_kqv_batch_seq_neq(d_embed, d_key):
    d_batch = d_seq = 2
    attn = Attention(d_embed=d_embed, d_key=d_key)
    embed = torch.rand(d_batch, d_seq, d_embed)
    for x in attn.compute_kqv(embed):
        # seq neq
        for batch_idx in range(2):
            assert not torch.allclose(x[batch_idx, 0], x[batch_idx, 1])
        # batch neq
        for seq_idx in range(2):
            assert not torch.allclose(x[0, seq_idx], x[1, seq_idx])


@pytest.mark.parametrize("d_key", range(1, 4))
def test_attention_prob(d_key):
    d_batch = 2
    d_seq = 2
    k = torch.rand(d_batch, d_seq, d_key)
    q = torch.rand(d_batch, d_seq, d_key)
    attn = Attention(d_embed=1, d_key=d_key)
    prob = attn.compute_prob(q, k)
    prob_summed = torch.sum(prob, dim=2)
    # sums to 1
    assert torch.allclose(prob_summed, torch.ones_like(prob_summed))
    for batch_idx in range(d_batch):
        assert not torch.allclose(prob[batch_idx, 0], prob[batch_idx, 1])


@pytest.mark.parametrize("d_embed, d_key", itertools.product(range(1, 4), repeat=2))
def test_attention_seq_neq(d_embed, d_key):
    d_batch = 1
    d_seq = 2
    attn = Attention(d_embed=d_embed, d_key=d_key)
    # with small d_key logits are very small and softmax(logits) returns uniform distribution,
    # due to float32 precision
    attn.norm = 0.02**2
    embed = torch.rand(d_batch, d_seq, d_embed)
    attn_out = attn(embed)
    assert not torch.allclose(attn_out[0, 0], attn_out[0, 1])
