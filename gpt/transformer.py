import torch
from torch import nn
from torchtyping import TensorType

from gpt.decoder import Decoder
from gpt.linear_normal import LinearNormal
from gpt.types import ModelTensor, PadMaskTensor


class TransformerLM(nn.Module):
    """Turning sequences of tokens into distributions over vocabulary, using Transformer under the hood

    Positional embeddings are learned.
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_ctx: int,
        d_model: int,
        d_head: int,
        pad_idx: int,
        dropout_p: float,
    ):
        super().__init__()
        self.transformer = Transformer(
            vocab_size=vocab_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_ctx=n_ctx,
            d_model=d_model,
            d_head=d_head,
            dropout_p=dropout_p,
        )
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, n_ctx, d_model)))
        self.embedding_dropout = nn.Dropout(p=dropout_p)

    def forward(
        self, ctx: TensorType["batch", "ctx", torch.long]
    ) -> TensorType["batch", "ctx", "vocab_size", torch.float]:
        batch_size = ctx.size()[0]
        embedding = self.token_embedding(ctx) + self.position_embedding.expand(batch_size, -1, -1)
        pad_mask = ctx == self.pad_idx
        return self.transformer(self.embedding_dropout(embedding), pad_mask)


class Transformer(nn.Module):
    """A stack of decoders + linear, turning a sequence of embeddings into a sequence of distributions over vocab

    It returns distributions as logits.
    """

    def __init__(
        self, vocab_size: int, n_layers: int, n_heads: int, n_ctx: int, d_model: int, d_head: int, dropout_p: float
    ):
        super().__init__()
        self.decoders = nn.ModuleList(
            [
                Decoder(n_heads=n_heads, n_ctx=n_ctx, d_model=d_model, d_head=d_head, dropout_p=dropout_p)
                for _ in range(n_layers)
            ]
        )
        self.linear = LinearNormal(d_model, vocab_size, bias=True)

    def forward(self, embed: ModelTensor, pad_mask: PadMaskTensor) -> TensorType["batch", "ctx", "vocab_size"]:
        decoders_out = embed
        for decoder in self.decoders:
            decoders_out = decoder(decoders_out, pad_mask)
        # (batch, ctx, model) -> (batch, ctx, vocab_size)
        return self.linear(decoders_out)
