import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm


def collate_gt(list_of_samples, tokenizer, max_context):
    list_of_contexts = [it["context"] for it in list_of_samples]
    batched_gts = [sample["candidates"][0] for sample in list_of_samples]

    batched_context = tokenizer(
        list_of_contexts, return_tensors="pt", max_length=max_context, truncation=True, padding=True
    )

    return batched_context, batched_gts


def collate_candidates(list_of_samples, tokenizer, max_context, max_candidates):
    list_of_contexts = [it["context"] for it in list_of_samples]
    list_of_candidates = [it for sample in list_of_samples for it in sample["candidates"]]

    batched_context = tokenizer(
        list_of_contexts, return_tensors="pt", max_length=max_context, truncation=True, padding=True
    ).input_ids

    batched_candidates = tokenizer(
        list_of_candidates, return_tensors="pt", max_length=max_candidates, truncation=True, padding=True
    ).input_ids

    return batched_context, batched_candidates


def calculate_hits_ppl(model, dataloader, num_candidates: int, device: str):
    hits = []
    ppls = []

    model.eval()
    with torch.no_grad():
        for b_context, b_candidates in tqdm(dataloader, desc="Calculating perplexity and hits"):
            context_ids = b_context.repeat_interleave(num_candidates, dim=0).to(device)
            candidates_ids = b_candidates.to(device)

            target_ids = candidates_ids.clone()
            pad_mask = target_ids.eq(0)
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
    ppl = torch.concat(ppls).float().median().item()

    return np.round(hit * 100, 1), np.round(ppl, 2)


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


def calculate_f1(
    model,
    tokenizer,
    dataloader,
    max_gen_len,
    do_sample,
    num_beams,
    device,
):
    f1s = []

    model.eval()
    with torch.no_grad():
        for b_context, b_gts in tqdm(dataloader, desc="Calculating f1-score"):
            b_context = b_context.to(device)
            b_gts = b_gts

            reply_ids = model.generate(**b_context, do_sample=do_sample, num_beams=num_beams, max_length=max_gen_len)
            generated = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

            f1s += calc_f1_score(generated, b_gts)

    f1 = np.mean(f1s)

    return np.round(f1 * 100, 2)
