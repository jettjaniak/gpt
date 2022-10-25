from torch import nn

from gpt.attention import MultiHeadAttention
from gpt.linear_normal import LinearNormal
from gpt.types import ModelTensor, PadMaskTensor


class Decoder(nn.Module):
    """Multi-head attention, position-wise feed-forward and residual connections.

    FF with single hidden layer. Residuals with dropout and layer norm.
    """

    def __init__(self, n_heads: int, n_ctx: int, d_model: int, d_head: int, dropout_p: float):
        super().__init__()
        self.attn = MultiHeadAttention(n_heads, d_model, d_head, dropout_p)
        d_ff = 4 * d_model
        self.ff = nn.Sequential(
            LinearNormal(d_model, d_ff, bias=True), nn.GELU(), LinearNormal(d_ff, d_model, bias=True)
        )
        self.layer_norm_attn = nn.LayerNorm([n_ctx, d_model])
        self.layer_norm_mlp = nn.LayerNorm([n_ctx, d_model])
        self.residual_dropout = nn.Dropout(p=dropout_p)

    def forward(self, embed: ModelTensor, pad_mask: PadMaskTensor) -> ModelTensor:
        attn_out = self.attn(embed, pad_mask)
        add_norm_attn_out = self.layer_norm_attn(attn_out + self.residual_dropout(embed))
        mlp_out = self.ff(add_norm_attn_out)
        return self.layer_norm_mlp(mlp_out + self.residual_dropout(add_norm_attn_out))
