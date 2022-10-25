import torch
from torchtyping import TensorType

HeadTensor = TensorType["batch", "ctx", "head", torch.float]
ModelTensor = TensorType["batch", "ctx", "model", torch.float]
PadMaskTensor = TensorType["batch", "ctx", torch.bool]
FullMaskTensor = TensorType["batch", "ctx", "ctx", torch.bool]
