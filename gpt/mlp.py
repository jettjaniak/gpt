import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


class MLP(nn.Module):
    """GELU activations, output linear"""

    def __init__(self, n_layers: int, layer_width: int):
        super().__init__()
        layers = []
        if n_layers > 0:
            layers = [nn.Linear(layer_width, layer_width)]
        for _ in range(n_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(layer_width, layer_width))
        self.layers = nn.Sequential(*layers)

    @typechecked
    def forward(self, input_: TensorType[..., "input_"]) -> TensorType[..., "input_"]:
        return self.layers(input_)
