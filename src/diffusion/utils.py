import torch
import torch.nn.functional as F

from src.diffusion.schedules import cosine_beta_schedule
from src.diffusion.schedules import linear_beta_schedule
from src.diffusion.schedules import sqrt_beta_schedule


def flat_mean(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


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

    sigma_0 = torch.sqrt(betas[0])

    return alphas_cumprod_prev, sigma_0


def get_xt(x_0, alphas_cumprod_prev, t):
    alphas_cumprod_prev_t = alphas_cumprod_prev.to(x_0.device)[t].reshape(-1, 1, 1)

    noise = torch.normal(0, 1, size=x_0.shape).to(x_0.device)
    x_t = torch.sqrt(alphas_cumprod_prev_t) * x_0 + torch.sqrt(1 - alphas_cumprod_prev_t) * noise

    return x_t


def prepare_x0(emb: torch.Tensor, sigma_0: torch.Tensor):
    noise = torch.normal(0, sigma_0, size=emb.shape).to(emb.device)
    x_0 = emb + noise

    return x_0


def get_diffusion_variables(
    diffusion_steps: int,
    emb: torch.Tensor,
    alphas_cumprod_prev: torch.Tensor,
    sigma_0: torch.Tensor,
):
    x_0 = prepare_x0(emb, sigma_0)

    t = torch.randint(1, diffusion_steps + 1, size=(x_0.shape[0],)).to(x_0.device)

    x_t = get_xt(x_0, alphas_cumprod_prev, t)

    return x_0, x_t, t
