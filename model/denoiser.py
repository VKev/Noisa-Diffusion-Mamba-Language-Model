import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from mamba_ssm import Mamba2
from torchinfo import summary

try:
    from model.sit import SiTBlock, FinalLayer
    from model.attention import SelfAttention
except ImportError:
    from sit import SiTBlock, FinalLayer
    from attention import SelfAttention


class SiTDenoiser(nn.Module):
    """
    Denoiser model replacing UNet with a stack of SiT blocks.
    """
    def __init__(
        self,
        hidden_size=768,
        num_timesteps=1000,
        num_layers=6,
        dropout=0.2,
        heads=8
    ):
        super().__init__()
        # Time-step embedding
        self.time_embedding = nn.Embedding(num_timesteps, hidden_size)
        self.time_pos_encoding = PositionalEncoding1D(hidden_size)
        # Stack of SiT blocks
        self.blocks = nn.ModuleList([
            SiTBlock(
                hidden_size=hidden_size,
                num_heads=heads,
                mlp_ratio=4.0,
                drop=dropout
            ) for _ in range(num_layers)
        ])
        # Final projection back to hidden_size
        self.final = FinalLayer(hidden_size, hidden_size)

    def forward(self, zt, t):
        """
        Args:
            zt: (B, N, hidden_size) noisy input
            t: LongTensor of shape (B,) timesteps
        """
        # Embed and encode timesteps
        te = self.time_embedding(t).unsqueeze(1)   # (B, 1, hidden_size)
        te = self.time_pos_encoding(te)           # (B, 1, hidden_size)
        c = te.squeeze(1)                         # (B, hidden_size)
        # Add time embedding
        x = zt + te
        # Pass through SiT blocks
        for block in self.blocks:
            x = block(x, c)
        # Final adaptive projection
        x = self.final(x, c)
        return x


if __name__ == "__main__":
    batch_size, seq_len, dim = 4, 1024, 768
    num_steps = 1000
    model = SiTDenoiser(
        hidden_size=dim,
        num_timesteps=num_steps,
        num_layers=4,
        dropout=0.1,
        heads=8
    )
    sample = torch.randn(batch_size, seq_len, dim)
    timesteps = torch.randint(0, num_steps, (batch_size,), dtype=torch.long)
    out = model(sample, timesteps)
    print(out.shape)  # Expected: (4, 10, 768)
