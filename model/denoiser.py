import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from mamba_ssm import Mamba2
from torchinfo import summary

try:
    from model.attention import SelfAttention
except ImportError:
    from attention import SelfAttention


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, heads=2):
        super(DownBlock, self).__init__()
        self.attn = SelfAttention(
            ScaledDotProductAttention(
                d_model=in_dim,
                d_k=in_dim,
                d_v=in_dim,
                h=heads,
                dropout=dropout
            )
        )
        self.act = nn.SiLU()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.act(x)
        x = self.linear(x)
        x = self.norm(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, heads=2):
        super(UpBlock, self).__init__()
        self.attn = SelfAttention(
            ScaledDotProductAttention(
                d_model=in_dim,
                d_k=in_dim,
                d_v=in_dim,
                h=heads,
                dropout=dropout
            )
        )
        self.act = nn.SiLU()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, skip=None, proj=None):
        x = self.attn(x)
        x = self.act(x)
        x = self.linear(x)
        x = self.norm(x)
        if skip is not None and proj is not None:
            x = x + proj(skip)
        return x


class Denoiser(nn.Module):
    def __init__(
        self,
        downsample=(512, 256, 128),
        in_dim=768,
        num_timesteps=1000,
        dropout=0.2,
        heads=2
    ):
        super(Denoiser, self).__init__()

        # Time-step embedding
        # Map discrete timestep to a vector of dimension in_dim
        self.time_embedding = nn.Embedding(num_timesteps, in_dim)
        # Optionally add sinusoidal positional encoding to the time embedding
        self.time_pos_encoding = PositionalEncoding1D(in_dim)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        dims = [in_dim] + list(downsample)
        for i in range(len(downsample)):
            self.down_blocks.append(
                DownBlock(
                    in_dim=dims[i],
                    out_dim=dims[i+1],
                    dropout=dropout,
                    heads=heads
                )
            )

        # Upsampling path
        upsample = list(reversed(downsample))
        up_out_dims = upsample[1:] + [dims[0]]

        self.up_blocks = nn.ModuleList()
        for i in range(len(up_out_dims)):
            self.up_blocks.append(
                UpBlock(
                    in_dim=upsample[i],
                    out_dim=up_out_dims[i],
                    dropout=dropout,
                    heads=heads
                )
            )

        # Skip connection projections
        self.skip_projs = nn.ModuleList()
        for skip_dim, out_dim in zip(upsample, up_out_dims):
            if skip_dim != out_dim:
                self.skip_projs.append(nn.Linear(skip_dim, out_dim))
            else:
                self.skip_projs.append(nn.Identity())

    def forward(self, zt, t):
        """
        Args:
            zt: Tensor of shape (batch, seq_len, in_dim), noisy input
            t: LongTensor of shape (batch,), timestep indices
        """
        # Embed timesteps and broadcast across sequence length
        te = self.time_embedding(t)              # (batch, in_dim)
        te = te.unsqueeze(1)                     # (batch, 1, in_dim)
        te = self.time_pos_encoding(te)          # add positional encoding
        # Now te is (batch, 1, in_dim)

        # Add time embedding to input
        x = zt + te                              # broadcast over seq_len

        # Encoder (downsampling)
        skips = []
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        # Decoder (upsampling)
        for up, proj, skip in zip(self.up_blocks, self.skip_projs, reversed(skips)):
            x = up(x, skip=skip, proj=proj)

        return x


if __name__ == "__main__":
    batch_size, seq_len, dim = 4, 10, 768
    num_steps = 1000
    model = Denoiser(in_dim=dim, num_timesteps=num_steps)
    sample = torch.randn(batch_size, seq_len, dim)
    timesteps = torch.randint(0, num_steps, (batch_size,), dtype=torch.long)
    out = model(sample, timesteps)
    print(out.shape)
