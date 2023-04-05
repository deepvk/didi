import torch

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
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    sigmas = torch.cat([sigmas.new_zeros([1]), sigmas])

    return sigmas


def scale_input(input, sigma_t):
    return input / (sigma_t**2 + 1) ** 0.5


def get_diffusion_variables(diffusion_steps: int, x_0: torch.Tensor, sigmas: torch.Tensor, noise: torch.Tensor = None):
    if noise is None:
        noise = torch.normal(0, 1, size=x_0.shape, device=x_0.device)

    t = torch.randint(1, diffusion_steps + 1, size=(x_0.shape[0], 1), device=x_0.device)
    sigma_t = sigmas[t].view(-1, 1, 1)
    x_t = scale_input(x_0 + sigma_t * noise, sigma_t)
    return x_t, t


def get_euler_variables(x_t, noise, sigma_t, s_churn, s_tmin, s_tmax, num_sigmas):
    noise.normal_(0, 1)

    gamma = min(s_churn / num_sigmas, 2**0.5 - 1) if s_tmin <= sigma_t <= s_tmax else 0.0
    sigma_hat = sigma_t * (gamma + 1)

    if gamma > 0:
        x_t = x_t + (sigma_hat**2 - sigma_t**2) ** 0.5 * noise
    return x_t, sigma_hat
