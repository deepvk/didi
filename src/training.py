import torch
import wandb
from tqdm import tqdm

from src.diffusion.utils import configure_schedule
from src.diffusion.utils import flat_mean
from src.diffusion.utils import get_diffusion_variables
from src.metrics import calculate_batch_ce
from src.metrics import calculate_ce


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    schedule: str,
    num_steps: int,
    device: str,
    project_name: str = "didi",
    logging_step: int = 100,
    step_freq: int = 10,
):
    wandb.init(project=project_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    alphas_cumprod_prev = configure_schedule(model.diffusion_steps, schedule).to(device)

    model.train()
    with tqdm(total=num_steps) as pbar:
        while num_steps:
            # b_context: [batch size, context seq len], b_gt: [batch size, target seq len]
            for b_context, b_gt in train_dataloader:
                context = b_context.to(device)
                gt = b_gt.to(device)

                emb = model.emb(gt)

                x_0, x_t, t = get_diffusion_variables(model.diffusion_steps, emb, alphas_cumprod_prev, device)

                x_0_hat = model(
                    encoder_input_ids=context,
                    decoder_inputs_embeds=x_t,
                    time_ids=t,
                )

                optimizer.zero_grad()

                pad_mask = gt == model.emb.padding_idx
                ce = calculate_batch_ce(model, x_0, gt, pad_mask)

                emb_mask = ~pad_mask.unsqueeze(-1)

                mse = torch.where(
                    t == 1,
                    flat_mean((x_0_hat - emb) ** 2 * emb_mask),
                    flat_mean((x_0_hat - x_0) ** 2 * emb_mask),
                ).mean()

                t0_loss = (x_0**2 * emb_mask).mean()

                wandb.log(
                    {"train_mse": mse.item(), "train_ce": ce.item(), "train_t0": t0_loss.item()}, step=logging_step
                )

                loss = mse + ce + t0_loss

                loss.backward()
                optimizer.step()

                pbar.update(1)
                num_steps -= 1

                if not num_steps:
                    break

            wandb.log({"val_ce": calculate_ce(model, val_dataloader, alphas_cumprod_prev, step_freq, device)})

    return model
