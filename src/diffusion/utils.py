import torch
import torch.nn.functional as F

from schedules import cosine_beta_schedule
from schedules import linear_beta_schedule


def flat_mean(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def configure_schedule(steps: int, schedule: str):
    if schedule == "linear":
        betas = linear_beta_schedule(steps)
    elif schedule == "cosine":
        betas = cosine_beta_schedule(steps)
    else:
        raise NotImplementedError(f"TODO: implement {schedule} schedule")

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod, (1, 0), value=1.0)
    return alphas_cumprod_prev


def get_xt(x_0, alphas_cumprod_prev, t, device):
    alphas_cumprod_prev_t = alphas_cumprod_prev[t].reshape(-1, 1, 1)

    noise = torch.normal(0, 1, size=x_0.shape).to(device)
    x_t = torch.sqrt(alphas_cumprod_prev_t) * x_0 + torch.sqrt(1 - alphas_cumprod_prev_t) * noise

    return x_t


def prepare_x0(emb, device):
    sigma_0 = 0.1

    noise = torch.normal(0, sigma_0, size=emb.shape).to(device)
    x_0 = emb + noise

    return x_0


def get_diffusion_variables(
    diffusion_steps,
    emb,
    alphas_cumprod_prev: torch.Tensor,
    device: str,
):
    x_0 = prepare_x0(emb, device)

    t = torch.randint(1, diffusion_steps + 1, size=(x_0.shape[0],)).to(device)

    x_t = get_xt(x_0, alphas_cumprod_prev, t, device)

    return x_0, x_t, t
