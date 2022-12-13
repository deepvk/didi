import torch


def linear_beta_schedule(steps: int, low: float = 1e-4, high: float = 0.02):
    return torch.linspace(low, high, steps)


def cosine_beta_schedule(steps: int, s: float = 0.008):
    x = torch.arange(steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 1 - 1e-4)
