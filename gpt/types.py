import torch
from torchtyping import TensorType

HeadTensor = TensorType["batch", "ctx", "head", torch.float]
ModelTensor = TensorType["batch", "ctx", "model", torch.float]
MaskTensor = TensorType["batch", "ctx", "ctx", torch.bool]
