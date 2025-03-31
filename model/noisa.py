import torch
import torch.nn as  nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
try:
    from model.attention import SelfAttention
except ImportError:
    from attention import SelfAttention

torch.set_float32_matmul_precision('high')

class Noisa(nn.Module):
    def __init__(self, vocab_size=50260, embed_dim=512, num_heads=4, dropout=0.1):
        super(Noisa, self).__init__()
        self.tokenEmbed = nn.Embedding(vocab_size, embed_dim)
        self.posEmbed = Summer(PositionalEncoding1D(embed_dim))
        self.selfAttn1 = SelfAttention(
            ScaledDotProductAttention(d_model=embed_dim, d_k=embed_dim, d_v=embed_dim, h=num_heads, dropout=dropout)
        )
        self.fc_0 = nn.Linear(embed_dim, embed_dim)
        self.selfAttn2 = SelfAttention(
            ScaledDotProductAttention(d_model=embed_dim, d_k=embed_dim, d_v=embed_dim, h=num_heads, dropout=dropout)
        )
        self.fc_1 = nn.Linear(embed_dim, embed_dim*4)
        self.fc_2 = nn.Linear(embed_dim*4, embed_dim*2)
        self.fc_out = nn.Linear(embed_dim*2, vocab_size)
        
        
    def forward(self, x, attention_mask=None):
        x = self.tokenEmbed(x)
        x = self.posEmbed(x)
        x = self.selfAttn1(x, attention_mask)
        x = F.relu(self.fc_0(x))
        x = self.selfAttn2(x, attention_mask)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        logits = self.fc_out(x)
        return logits
    
if __name__ == "__main__":
    model = Noisa().to('cuda')
    x = torch.randint(low=0, high=50258, size=(2, 768)).to('cuda')
    seq_len = x.size(1)
    device = x.device
    attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    output = model(x, attention_mask)
    print(output.shape)     
    