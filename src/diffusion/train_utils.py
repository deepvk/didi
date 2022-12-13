import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from schedules import cosine_beta_schedule
from schedules import linear_beta_schedule


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


def get_diffusion_variables(
    model,
    gt,
    max_gen: int,
    diffusion_steps: int,
    alphas_cumprod_prev: torch.Tensor,
    device: str,
):
    pad_emb = model.get_embeddings(torch.LongTensor([0]))

    x_0 = model.get_embeddings(gt.input_ids)
    padding = pad_emb.repeat((x_0.shape[0], max_gen - x_0.shape[1], 1))
    x_0 = torch.concat((x_0, padding), dim=1)

    t = torch.randint(diffusion_steps, size=(1,))

    x_t = torch.sqrt(alphas_cumprod_prev[t]) * x_0 + torch.sqrt(1 - alphas_cumprod_prev[t]) * torch.normal(
        0, 1, size=x_0.shape
    )

    return x_0.to(device), x_t.to(device), t.to(device)


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
    tokenizer,
    dataloader,
    schedule: str,
    diffusion_steps: int,
    num_epochs: int,
    max_context: int,
    max_gen: int,
    device: str,
    project_name: str = "didi",
    run_name: str = "Training diffusion",
):
    wandb.init(project=project_name, name=run_name)

    optimizer = torch.optim.Adam(model.parameters())

    alphas_cumprod_prev = configure_schedule(diffusion_steps, schedule)

    for _ in tqdm(range(num_epochs), desc="Training"):
        for batch in tqdm(dataloader, desc="Epoch"):
            context = tokenizer(
                batch["context"],
                return_tensors="pt",
                max_length=max_context,
                truncation=True,
                padding=True,
            ).to(device)
            gt = tokenizer(
                batch["candidates"][0],
                return_tensors="pt",
                max_length=max_gen,
                truncation=True,
                padding=True,
            )

            x_0, x_t, t = get_diffusion_variables(model, gt, max_gen, diffusion_steps, alphas_cumprod_prev, device)

            x_0_hat = model(
                encoder_input_ids=context.input_ids,
                encoder_attention_mask=context.attention_mask,
                decoder_inputs_embeds=x_t,
                time_ids=t,
            )

            optimizer.zero_grad()

            loss = F.mse_loss(x_0_hat, x_0)

            wandb.log({"loss": loss.item()})

            loss.backward()
            optimizer.step()

    return model
