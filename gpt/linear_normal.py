import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


class LinearNormal(nn.Linear):
    """N(0, 0.02) weight initialization, bias default (if used)"""

    def __init__(self, in_features: int, out_features: int, bias: bool, **kwargs):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, **kwargs)
        nn.init.normal_(self.weight, mean=0, std=0.02)
