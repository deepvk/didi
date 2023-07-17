from functools import partial

import torch
from math import sqrt

from src.diffusion.utils import scale_input
from src.metrics import calculate_batch_ce


def freeze_params(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def flat_mean(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def rsqrt_with_warmup(step: int, max_lr: float, min_lr: float, warmup: int) -> float:
    """Scheduler for learning rate with a form of reverse sqrt (known as Noam favorite scheduler):
        `lr_t = max_lr * sqrt(1 / t)`

    Warm-up increases learning rate from 0 with square root form and then smoothly decay with reverse square root.
        `lr_t = max_lr * sqrt(t / warmup)` if t <= warmup
        `lr_t = max_lr * sqrt(warmup / t)` if t > warmup

    Also, there is control of minimum learning rate

    :param step: current step
    :param max_lr: maximum learning rate
    :param min_lr: minimum learning rate
    :param warmup: number of warmup steps
    :return: next learning rate
    """
    if warmup != 0 and step < warmup:
        return max_lr * sqrt(step / warmup)

    if warmup == 0:
        lr = max_lr * sqrt(1 / step)
    else:
        lr = max_lr * sqrt(warmup / step)

    if min_lr is not None:
        lr = max(lr, min_lr)
    return lr


def get_cached_content(model, encoder_input_ids, encoder_attention_mask):
    with torch.no_grad():
        context = model.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask).last_hidden_state

        if model.encoder_dim != model.decoder_dim:
            context = model.adapter(context)
    return context


def get_optimizers(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)  # Fully control LR from scheduler
    scheduler_lambda = partial(rsqrt_with_warmup, max_lr=model.lr, min_lr=model.min_lr, warmup=model.warmup)
    lr_scheduler_config = {
        "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda),
        "interval": "step",
    }
    return [optimizer], [lr_scheduler_config]


def calculate_train_step(model, emb, x_0, x_0_hat, target, t):
    logits = model.classifier(x_0)  # [batch size; seq len; vocab size]
    ce = calculate_batch_ce(logits, target.input_ids, target.attention_mask)

    non_pad_mask = target.attention_mask.unsqueeze(-1)
    mse = torch.where(
        t == 1,
        flat_mean((x_0_hat - emb) ** 2 * non_pad_mask),
        flat_mean((x_0_hat - x_0) ** 2 * non_pad_mask),
    ).mean()

    noise, sigma_T = torch.randn_like(x_0), model.sigmas[-1]
    x_T = scale_input(x_0 + sigma_T * noise, sigma_T)
    t_T_loss = (x_T**2 * non_pad_mask).mean()

    loss = mse + ce + t_T_loss

    with torch.no_grad():
        logits_hat = model.classifier(x_0_hat)
        ce_hat = calculate_batch_ce(logits_hat, target.input_ids, target.attention_mask)

    metrics = {"train/mse": mse, "train/ce": ce, "train/t_T": t_T_loss, "train/loss": loss, "train/ce_hat": ce_hat}
    return loss, metrics
