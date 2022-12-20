import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from schedules import cosine_beta_schedule
from schedules import linear_beta_schedule
from src.metrics import calculate_ce


def configure_schedule(steps: int, schedule: str):
    if schedule == "linear":
        betas = linear_beta_schedule(steps)
    elif schedule == "cosine":
        betas = cosine_beta_schedule(steps)
    else:
        raise NotImplementedError(f"TODO: implement {schedule} schedule")

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    return alphas_cumprod_prev


def get_xt(x_0, alphas_cumprod_prev, t, device):
    x_t = torch.sqrt(alphas_cumprod_prev[t]) * x_0 + (t > 0) * torch.sqrt(1 - alphas_cumprod_prev[t]) * torch.normal(
        0, 1, size=x_0.shape
    ).to(device)

    return x_t


def prepare_x0(model, gt, max_gen, device):
    pad_emb = model.pad_embedding

    x_0 = model.embeddings(gt.input_ids)
    padding = pad_emb.repeat((x_0.shape[0], max_gen - x_0.shape[1], 1)).to(device)
    x_0 = torch.concat((x_0, padding), dim=1)

    return x_0


def get_diffusion_variables(
    model,
    gt,
    max_gen: int,
    alphas_cumprod_prev: torch.Tensor,
    device: str,
):
    x_0 = prepare_x0(model, gt, max_gen, device)

    t = torch.randint(model.diffusion_steps, size=(1,)).to(device)

    x_t = get_xt(x_0, alphas_cumprod_prev, t, device)

    return x_0, x_t, t


def collate_gt(list_of_samples, tokenizer, max_context, max_gt):
    list_of_contexts = [it["context"] for it in list_of_samples]
    batched_gts = [sample["candidates"][0] for sample in list_of_samples]

    batched_context = tokenizer(
        list_of_contexts, return_tensors="pt", max_length=max_context, truncation=True, padding=True
    )

    batched_gts = tokenizer(batched_gts, return_tensors="pt", max_length=max_gt, truncation=True, padding=True)

    return batched_context, batched_gts


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    schedule: str,
    num_steps: int,
    max_gen: int,
    device: str,
    project_name: str = "didi",
    logging_step: int = 10,
):
    wandb.init(project=project_name)

    optimizer = torch.optim.Adam(model.parameters())

    alphas_cumprod_prev = configure_schedule(model.diffusion_steps, schedule).to(device)

    model.train()
    with tqdm(total=num_steps) as pbar:
        while num_steps:
            for b_context, b_gt in tqdm(train_dataloader):
                context = b_context.to(device)
                gt = b_gt.to(device)

                x_0, x_t, t = get_diffusion_variables(model, gt, max_gen, alphas_cumprod_prev, device)

                x_0_hat, probs = model(
                    encoder_input_ids=context.input_ids,
                    encoder_attention_mask=context.attention_mask,
                    decoder_inputs_embeds=x_t,
                    time_ids=t,
                )

                optimizer.zero_grad()

                mse = F.mse_loss(x_0_hat, x_0)

                target = F.pad(b_gt.input_ids, (0, probs.shape[1] - b_gt.input_ids.shape[1]), "constant", -100)
                ce = F.cross_entropy(torch.transpose(probs, 1, 2), target)

                wandb.log({"train_mse": mse.item(), "train_ce": ce.item()}, step=logging_step)

                loss = mse + ce

                loss.backward()
                optimizer.step()

                pbar.update(1)
                num_steps -= 1

                if not num_steps:
                    break

            wandb.log({"val_ce": calculate_ce(model, val_dataloader, alphas_cumprod_prev, max_gen, device)})

    return model
