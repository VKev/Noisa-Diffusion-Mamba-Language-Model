import torch
import torch.nn as  nn
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention


class SelfAttention(nn.Module):
    def __init__(self, attention_module):
        super(SelfAttention, self).__init__()
        self.attn = attention_module
        self.norm = nn.LayerNorm(attention_module.d_model)

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            batch_size, seq_len = x.shape[:2]
            h = self.attn.h
            attention_mask = attention_mask.bool()
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                attention_mask = attention_mask.expand(-1, h, seq_len, -1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1).expand(-1, h, -1, -1)

            attn_out = self.attn(x, x, x, attention_mask=attention_mask)
        else:
            attn_out = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm(x)
        return x