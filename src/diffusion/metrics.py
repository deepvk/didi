import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from src.sampling import sample


@torch.no_grad()
def calculate_ppl(model, dataloader, pad_idx, config, device):
    """Calculate and perplexity of the model on the given dataset."""
    ppls = []

    model.eval()
    logger.info("Calculating perplexity")
    for raw_context, target in tqdm(dataloader, desc="Calculating perplexity"):
        raw_context = raw_context.to(device)
        target = target.to(device)

        candidates_ids = target.input_ids

        target_ids = candidates_ids.clone()
        pad_mask = target_ids.eq(pad_idx)
        target_ids[pad_mask] = -100

        pred_logits = sample(raw_context, model, None, "ddpm", config.didi.step_freq, device, raw_output=True)

        shift_labels = target_ids[..., 1:].contiguous()
        shift_logits = pred_logits[..., 1 : shift_labels.shape[1] + 1, :].contiguous()
        shift_attention_mask_batch = pad_mask[..., 1:].contiguous().eq(0)

        perplexity_batch = torch.exp(
            (
                F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, reduction="none")
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls.append(perplexity_batch)

    ppl = torch.concat(ppls).float().mean().item()
    return ppl
