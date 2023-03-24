import torch
import torch.nn.functional as F

from src.diffusion.schedules import cosine_beta_schedule, linear_beta_schedule, sqrt_beta_schedule


def configure_schedule(steps: int, schedule: str):
    if schedule == "linear":
        betas = linear_beta_schedule(steps)
    elif schedule == "cosine":
        betas = cosine_beta_schedule(steps)
    elif schedule == "sqrt":
        betas = sqrt_beta_schedule(steps)
    else:
        raise NotImplementedError(f"TODO: implement {schedule} schedule")

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod, (1, 0), value=1.0)

    sigma_0 = torch.sqrt(betas[0]).item()

    return alphas_cumprod_prev, sigma_0


def get_xt(x_0, alphas_cumprod_prev, t, noise=None):
    if noise is None:
        noise = torch.normal(0, 1, size=x_0.shape, device=x_0.device)

    alphas_cumprod_prev_t = alphas_cumprod_prev[t].view(-1, 1, 1)
    x_t = torch.sqrt(alphas_cumprod_prev_t) * x_0 + torch.sqrt(1 - alphas_cumprod_prev_t) * noise
    return x_t


def prepare_x0(emb: torch.Tensor, sigma_0: float):
    noise = torch.normal(0, sigma_0, size=emb.shape, device=emb.device)
    x_0 = emb + noise
    return x_0


def get_diffusion_variables(diffusion_steps: int, emb: torch.Tensor, alphas_cumprod_prev: torch.Tensor, sigma_0: float):
    x_0 = prepare_x0(emb, sigma_0)
    t = torch.randint(1, diffusion_steps + 1, size=(x_0.shape[0],), device=x_0.device)
    x_t = get_xt(x_0, alphas_cumprod_prev, t)
    return x_0, x_t, t
