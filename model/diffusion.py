# import torch
# from diffusers import LMSDiscreteScheduler

# B, N, D = 5, 256, 768 
# device = "cuda" if torch.cuda.is_available() else "cpu"

# batch_tokens = torch.randn(B, N, D).to(device)
# scheduler = LMSDiscreteScheduler(
#     beta_start=0.00085,
#     beta_end=0.012,
#     beta_schedule="scaled_linear",
#     num_train_timesteps=1000
# )
# scheduler.set_timesteps(1000)

