import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm


def calculate_hits_ppl(model, tokenizer, dataset, max_len: int, device: str):
    hits = []
    ppl = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(dataset, desc="Calculating perplexity and hits"):
            context = data["context"]
            candidates = data["candidates"]

            input_ids = (
                tokenizer(
                    context,
                    return_tensors="pt",
                    max_length=max_len,
                    truncation=True,
                    padding=True,
                )
                .input_ids.repeat(len(candidates), 1)
                .to(device)
            )

            out_ids = tokenizer(
                candidates,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
                padding=True,
            ).input_ids.to(device)

            target_ids = out_ids.clone()
            pad_mask = target_ids.eq(0)
            target_ids[pad_mask] = -100

            model_output = model(input_ids, decoder_input_ids=out_ids, labels=target_ids)

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
            )

            hits.append(perplexity_batch.argmin().eq(0).item())
            ppl.append(perplexity_batch[0].item())

    return np.round(np.mean(hits) * 100, 1), np.round(np.mean(ppl), 2)


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
    max_context_len,
    max_gen_len,
    do_sample,
    num_beams,
    device,
):
    f1 = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Calculating f1-score"):
            context = data["context"]
            candidates = data["candidates"]

            gts = [batch[0] for batch in candidates]

            input = tokenizer(
                context,
                return_tensors="pt",
                max_length=max_context_len,
                truncation=True,
                padding=True,
            ).to(device)

            reply_ids = model.generate(**input, do_sample=do_sample, num_beams=num_beams, max_length=max_gen_len)
            generated = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

            f1 += calc_f1_score(generated, gts)

    return np.round(np.mean(f1) * 100, 2)
