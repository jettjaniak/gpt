import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from gpt.decoder import Decoder
from gpt.attention import EmbedTensor

patch_typeguard()


# TODO: dropout everywhere
# TODO: padding mask
# TODO: peeking mask


class TransformerLM(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        n_decoders: int,
        n_decoder_mlp_layers: int,
        n_attn_heads: int,
        seq_len: int,
        d_embed: int,
        d_key: int,
    ):
        super().__init__()
        self.transformer = Transformer(
            n_tokens=n_tokens,
            n_decoders=n_decoders,
            n_decoder_mlp_layers=n_decoder_mlp_layers,
            n_attn_heads=n_attn_heads,
            seq_len=seq_len,
            d_embed=d_embed,
            d_key=d_key,
        )
        self.token_embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_embed, padding_idx=self.pad_idx)
        self.position_embedding = nn.Embedding(num_embeddings=seq_len, embedding_dim=d_embed)
        self._position_idx = torch.range(0, seq_len)

    def forward(self, sequence: TensorType["batch", "seq", torch.long]) -> TensorType["batch", "tokens"]:
        d_batch, d_seq = sequence.size()
        position_idx = self._position_idx.expand(d_batch, d_seq)
        embed = self.token_embedding(sequence) + self.position_embedding(position_idx)
        return self.transformer(embed)


class Transformer(nn.Module):
    """A stack of decoders + softmax, turning sequences of embeddings into distribution over tokens"""

    def __init__(
        self,
        n_tokens: int,
        n_decoders: int,
        n_decoder_mlp_layers: int,
        n_attn_heads: int,
        seq_len: int,
        d_embed: int,
        d_key: int,
    ):
        super().__init__()
        self.decoders = nn.Sequential(
            *[
                Decoder(
                    n_mlp_layers=n_decoder_mlp_layers,
                    n_heads=n_attn_heads,
                    seq_len=seq_len,
                    d_embed=d_embed,
                    d_key=d_key,
                )
                for _ in range(n_decoders)
            ]
        )
        self.linear = nn.Linear(seq_len * d_embed, n_tokens)

    @typechecked
    def forward(self, embed: EmbedTensor) -> TensorType["batch", "tokens"]:
        # -> (batch, seq, embed)
        decoders_out = self.decoders(embed)
        # We present full sequences to Linear + Softmax
        # -> (batch, seq * embed)
        d_batch, d_seq, d_embed = embed.size()
        decoders_out_2d = decoders_out.view(d_batch, d_seq * d_embed)
        # -> (batch, d_token)
        linear_out = self.linear(decoders_out_2d)
        return torch.softmax(linear_out, dim=1)
