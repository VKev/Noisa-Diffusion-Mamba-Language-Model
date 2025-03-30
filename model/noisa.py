import torch
import torch.nn as  nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
try:
    from model.attention import SelfAttention
except ImportError:
    from attention import SelfAttention

class Noisa(nn.Module):
    def __init__(self, vocab_size=50258, embed_dim=512, num_heads=4, dropout=0.2):
        super(Noisa, self).__init__()
        self.tokenEmbed = nn.Embedding(vocab_size, embed_dim)
        self.posEmbed = Summer(PositionalEncoding1D(embed_dim))
        self.selfAttn = SelfAttention(
            ScaledDotProductAttention(d_model=embed_dim, d_k=embed_dim, d_v=embed_dim, h=num_heads, dropout=dropout)
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x, attention_mask=None):
        x = self.tokenEmbed(x)
        x = self.posEmbed(x)
        x = self.selfAttn(x, attention_mask)
        x = F.relu(x)
        logits = self.fc_out(x)
        return logits
    
if __name__ == "__main__":
    model = Noisa().to('cuda')
    x = torch.randint(low=0, high=50258, size=(8, 2024)).to('cuda')
    seq_len = x.size(1)
    device = x.device
    attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    output = model(x, attention_mask)
    print(output.shape)     
    