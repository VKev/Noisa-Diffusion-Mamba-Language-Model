import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from mamba_ssm import Mamba2
from torchinfo import summary
from diffusers import DDPMScheduler

try:
    from model.vae import TokenVAE
    from model.attention import SelfAttention
    from model.denoiser import Denoiser
except ImportError:
    from vae import TokenVAE
    from denoiser import Denoiser
    from attention import SelfAttention

torch.set_float32_matmul_precision('high')

class Noisa(nn.Module):
    def __init__(
        self,
        vocab_size=50260,
        embed_dim=2048,
        encode_dims=[1512, 1024, 768],
        downsample=[512, 256, 128],
        num_timesteps=1000,
        dropout=0.2
    ):
        super(Noisa, self).__init__()
        # Embeddings
        self.tokenEmbed = nn.Embedding(vocab_size, embed_dim)
        self.posEmbed = Summer(PositionalEncoding1D(embed_dim))
        self.selfAttn = SelfAttention(
            ScaledDotProductAttention(
                d_model=embed_dim,
                d_k=embed_dim,
                d_v=embed_dim,
                h=4,
                dropout=dropout
            )
        )
        # VAE encoder & latent projections
        self.vae = TokenVAE(input_dim=embed_dim, encode_dims=encode_dims, dropout=dropout)
        self.fc_mu = nn.Linear(encode_dims[-1], encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-1], encode_dims[-1])
        # Denoiser
        self.num_timesteps = num_timesteps
        self.denoiser = Denoiser(
            downsample=downsample,
            in_dim=encode_dims[-1],
            num_timesteps=num_timesteps,
            dropout=dropout
        )
        # Output
        self.output_linear = nn.Linear(embed_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, attention_mask=None):
        x = self.tokenEmbed(x)
        x = self.posEmbed(x)
        x = self.selfAttn(x, attention_mask)
        x = self.vae.encode(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def denoise(self, zt, t):
        return self.denoiser(zt, t)

    def decode(self, z):
        x = self.vae.decode(z)
        logits = self.output_linear(x)
        return logits

    def forward(self, x, attention_mask=None):
        mu, logvar = self.encode(x, attention_mask)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Noisa().to(device)

    batch_size, seq_len = 2, 1024
    x = torch.randint(0, 50260, (batch_size, seq_len), device=device)
    attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)

    print(model)
    summary(model, input_data=(x, attention_mask), device=device)

    z, mu, logvar = model(x, attention_mask)
    print("Latent z shape:", z.shape)

    scheduler = DDPMScheduler(num_train_timesteps=model.num_timesteps)
    timesteps = torch.randint(0, model.num_timesteps, (batch_size,), dtype=torch.long, device=device)
    
    noise = torch.randn_like(z)
    noisy_z = scheduler.add_noise(z, noise, timesteps)

    pred_noise = model.denoise(noisy_z, timesteps)
    print("Predicted noise shape:", pred_noise.shape)

    logits = model.decode(z)
    print("Logits shape:", logits.shape)
