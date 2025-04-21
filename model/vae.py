import torch
import torch.nn as  nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from mamba_ssm import Mamba2
from torchinfo import summary
try:
    from model.attention import SelfAttention
except ImportError:
    from attention import SelfAttention
torch.set_float32_matmul_precision('high')

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(GatedMLP, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        x_norm = self.norm(x)
        x_proj = self.fc1(x_norm)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x_gated = self.act(x1) * x2
        x_gated = self.dropout(x_gated)
        x_out = self.fc2(x_gated)
        return self.residual_proj(x) + x_out
    

class TokenVAE(nn.Module):
    def __init__(self, 
                 input_dim=2048, 
                 encode_dims = [1512, 1024, 768],
                 dropout=0.2):
        super(TokenVAE, self).__init__()
        self.input_dim = input_dim
        encoder_layers = []
        for hidden_dim in encode_dims:
            encoder_layers.append(GatedMLP(input_dim, hidden_dim=hidden_dim, dropout=dropout))
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        input_dim = encode_dims[-1]
        for hidden_dim in reversed(encode_dims[:-1]):
            decoder_layers.append(
                GatedMLP(input_dim, hidden_dim=hidden_dim, dropout=dropout)
            )
            input_dim = hidden_dim
        decoder_layers.append(
            GatedMLP(input_dim, hidden_dim=self.input_dim, dropout=dropout)
        )
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        x = self.encoder(x)
        return x
        
    def decode(self, x):
        x = self.decoder(x)
        return x
        