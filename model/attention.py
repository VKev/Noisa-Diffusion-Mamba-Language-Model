import torch
import torch.nn as  nn
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention


class SelfAttention(nn.Module):
    def __init__(self, attention_module):
        super(SelfAttention, self).__init__()
        self.attn = attention_module
        self.norm = nn.LayerNorm(attention_module.d_model)

    def forward(self, x, attention_mask=None):
        assert attention_mask is not None, "attention_mask must be provided"
        attn_out = self.attn(x, x, x, attention_mask=attention_mask)
        x = x + attn_out
        x = self.norm(x)
        return x