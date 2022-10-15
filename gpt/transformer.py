import torch
from torch import nn
from torchtyping import TensorType

from gpt.decoder import Decoder
from gpt.linear_normal import LinearNormal
from gpt.types import ModelTensor, MaskTensor


class TransformerLM(nn.Module):
    """Turning sequences of tokens into distributions over vocabulary, using Transformer under the hood

    Positional embeddings are learned.
    """

    def __init__(
        self, vocab_size: int, n_layers: int, n_heads: int, n_ctx: int, d_model: int, d_head: int, dropout_p: float
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
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=self.pad_idx)
        # TODO: just use a matrix
        self.position_embedding = nn.Embedding(num_embeddings=n_ctx, embedding_dim=d_model)
        self._position_idx = torch.range(0, n_ctx)
        self.embedding_dropout = nn.Dropout(p=dropout_p)

    def forward(
        self, sequence: TensorType["batch", "ctx", torch.long], mask: MaskTensor
    ) -> TensorType["batch", "vocab_size"]:
        batch_size, n_ctx = sequence.size()
        position_idx = self._position_idx.expand(batch_size, n_ctx)
        embed = self.embedding_dropout(self.token_embedding(sequence) + self.position_embedding(position_idx))
        return self.transformer(embed, mask)


class Transformer(nn.Module):
    """A stack of decoders + softmax, turning sequences of embeddings into distributions over vocabulary"""

    def __init__(
        self, vocab_size: int, n_layers: int, n_heads: int, n_ctx: int, d_model: int, d_head: int, dropout_p: float
    ):
        super().__init__()
        decoder_kwargs = dict(n_heads=n_heads, n_ctx=n_ctx, d_model=d_model, d_head=d_head, dropout_p=dropout_p)
        self.first_decoder = Decoder(**decoder_kwargs)
        self.other_decoders = nn.Sequential(*[Decoder(**decoder_kwargs) for _ in range(n_layers - 1)])
        self.linear = LinearNormal(n_ctx * d_model, vocab_size, bias=True)

    def forward(self, embed: ModelTensor, mask: MaskTensor) -> TensorType["batch", "vocab_size"]:
        # -> (batch, ctx, model)
        first_decoder_out = self.first_decoder(embed, mask)  # only first decoder input should be masked
        other_decoders_out = self.other_decoders(first_decoder_out)
        # We present full sequences to Linear + Softmax
        # -> (batch, ctx * model)
        batch_size, n_ctx, d_model = embed.size()
        other_decoders_out_2d = other_decoders_out.view(batch_size, n_ctx * d_model)
        # -> (batch, vocab_size)
        linear_out = self.linear(other_decoders_out_2d)
        return torch.softmax(linear_out, dim=1)
