# import torch
# from mamba_ssm import Mamba2

# batch, length, dim = 2, 64, 256
# x = torch.randn(batch, length, dim).contiguous().to("cuda")

# model = Mamba2(
#     d_model=dim,  # Model dimension d_model
#     d_state=128,  # SSM state expansion factor, typically 64 or 128
#     d_conv=4,     # Local convolution width
#     expand=2,     # Block expansion factor
# ).to("cuda")

# y = model(x)
# assert y.shape == x.shape
