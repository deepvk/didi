import torch
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm

from src.diffusion.utils import get_xt
from src.diffusion.utils import prepare_x0


@torch.no_grad()
def calculate_hits_ppl(model, dataloader, pad_idx, device: str):
    """Calculate hits@1 and perplexity of the model on the given dataset."""
    hits = []
    ppls = []

    model.eval()
    logger.info("Calculating hits@1 and perplexity")
    for b_context, b_candidates in tqdm(dataloader, desc="Calculating perplexity and hits"):
        num_candidates = b_candidates.shape[1]
        context_ids = b_context.repeat_interleave(num_candidates, dim=0).to(device)
        candidates_ids = b_candidates.to(device).view(-1, b_candidates.shape[-1])

        target_ids = candidates_ids.clone()
        pad_mask = target_ids.eq(pad_idx)
        target_ids[pad_mask] = -100

        model_output = model(context_ids, decoder_input_ids=candidates_ids, labels=target_ids)

        pred_logits = model_output.logits

        shift_logits = pred_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        shift_attention_mask_batch = pad_mask[..., 1:].contiguous().eq(0)

        perplexity_batch = torch.exp(
            (
                F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, reduction="none")
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        ).view(-1, num_candidates)

        hits.append(perplexity_batch.argmin(dim=1).eq(0))
        ppls.append(perplexity_batch[:, 0])

    hit = torch.concat(hits).float().mean().item()
    ppl = torch.concat(ppls).float().mean().item()

    return hit * 100, ppl


def calc_f1_score(gens, gts):
    f1s = []

    for gen, gt in zip(gens, gts):
        gen = set(gen.split())
        gt = set(gt.split())

        precision = len(gen & gt) / len(gen)
        recall = len(gen & gt) / len(gt)

        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        f1s.append(f1)

    return f1s


@torch.no_grad()
def calculate_f1(model, tokenizer, dataloader, max_gen_len, do_sample, num_beams, device):
    """Calculate F1 score of the model on the given dataset."""
    f1s = []

    model.eval()
    logger.info("Calculating F1-score")
    for b_context, b_candidates in tqdm(dataloader, desc="Calculating F1-score"):
        b_context = b_context.to(device)
        b_gts = b_candidates[:, 0]

        ground_truth = tokenizer.batch_decode(b_gts, skip_special_tokens=True)

        reply_ids = model.generate(b_context, do_sample=do_sample, num_beams=num_beams, max_length=max_gen_len)
        generated = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

        f1s += calc_f1_score(generated, ground_truth)

    f1 = sum(f1s) / len(f1s)
    return f1 * 100


def calculate_batch_ce(model, x_0, gt, pad_mask):
    probs = model.classifier(x_0)

    target = gt.detach().clone()

    target[pad_mask] = -100
    ce = F.cross_entropy(torch.transpose(probs, 1, 2), target)

    return ce


@torch.no_grad()
def calculate_ce(model, val_dataloader, alphas_cumprod_prev, step_freq, device):
    ces = []

    model.eval()
    for b_context, b_gt in tqdm(val_dataloader, desc="Validating"):
        context = b_context.to(device)
        gt = b_gt.to(device)

        emb = model.emb(gt)
        x_0 = prepare_x0(emb, device)

        ones = torch.ones(x_0.shape[0]).long().to(device)

        for time in range(model.diffusion_steps, 1, -step_freq):
            t = ones * time

            x_t = get_xt(x_0, alphas_cumprod_prev, t, device)

            x_0 = model(
                encoder_input_ids=context,
                decoder_inputs_embeds=x_t,
                time_ids=t,
            )

        x_1 = get_xt(x_0, alphas_cumprod_prev, ones, device)

        x_0 = model(
            encoder_input_ids=context,
            decoder_inputs_embeds=x_1,
            time_ids=ones,
        )

        pad_mask = gt == model.emb.padding_idx

        ces.append(calculate_batch_ce(model, x_0, gt, pad_mask).item())

    return sum(ces) / len(ces)
