import torch
import random


def random_mask(batch_size, n_ctx):
    """With at least one True entry along last dimension"""
    mask = torch.rand(batch_size, n_ctx, n_ctx) < 0.8
    for i in range(n_ctx):
        if not torch.any(mask[:, i]):
            mask[:, i, random.randint(0, n_ctx - 1)] = True
    return mask


def random_embed_mask_equal_batch(n_ctx, d_model):
    """Random embed and mask, duplicated along batch dim"""
    single_embed = torch.rand(1, n_ctx, d_model)
    embed = torch.cat([single_embed, single_embed], dim=0)
    single_mask = random_mask(1, n_ctx)
    mask = torch.cat([single_mask, single_mask])
    return embed, mask
