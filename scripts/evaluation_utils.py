import argparse

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from tqdm import tqdm


def configure_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help="Seq2seq model name from huggingface")
    parser.add_argument("-p", "--path", help="Path to dataset for evaluation")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("-s", "--side", default="left", help="Side of truncation and padding for tokenizer")
    parser.add_argument("-cl", "--max_context", type=int, default=256, help="Max length of context fed into model")
    parser.add_argument("-gl", "--max_gen", type=int, default=64, help="Max length of generated output")
    parser.add_argument("-d", "--device", default="cpu", help="Device on which to evaluate the model")
    parser.add_argument("-ds", "--do_sample", default=True, help="parameter for generation method")
    parser.add_argument("-nb", "--num_beams", default=4, help="Parameter for generation method")

    return parser


class ConvAI2Dataset(Dataset):
    """ConvAI2 Dataset"""

    def __init__(self, path):
        """
        Args:
            path (str): File path to ConvAI2 dataset
        """

        persona = "your persona"
        context = ""

        candidates = []

        dataset = {"context": [], "candidates": []}

        with open(path, "r") as f:
            for line in f.readlines():
                if line[2 : 2 + len(persona)] == persona:
                    context = ""
                    candidates = []
                    continue

                else:
                    prev_candidates = candidates

                    utterance1, utterance2, _, candidates = line[2:].split("\t")
                    candidates = candidates.split("|")[:-1]

                    if prev_candidates:
                        dataset["context"].append(context)
                        dataset["candidates"].append([utterance1] + candidates)

                    context += "<s>" + utterance1 + "</s>" + "<s>" + utterance2 + "</s>"

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["context"])

    def __getitem__(self, idx):
        sample = {
            "context": self.dataset["context"][idx],
            "candidates": self.dataset["candidates"][idx],
        }

        return sample


def calculate_hits_ppl(model, tokenizer, dataset, max_len, device):
    hits = []
    ppl = []

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    model.eval()
    with torch.no_grad():
        for data in tqdm(dataset, desc="Calculating perplexity and hits"):
            context = data["context"]
            candidates = data["candidates"]

            input_ids = (
                tokenizer(context, return_tensors="pt", max_length=max_len, truncation=True, padding=True)
                .input_ids.repeat(len(candidates), 1)
                .to(device)
            )

            out_ids = tokenizer(
                candidates, return_tensors="pt", max_length=max_len, truncation=True, padding=True
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
                (loss_fn(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
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


def calculate_f1(model, tokenizer, dataloader, max_context_len, max_gen_len, do_sample, num_beams, device):
    f1 = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Calculating f1-score"):
            context = data["context"]
            candidates = data["candidates"]

            gts = [batch[0] for batch in candidates]

            input = tokenizer(
                context, return_tensors="pt", max_length=max_context_len, truncation=True, padding=True
            ).to(device)

            reply_ids = model.generate(**input, do_sample=do_sample, num_beams=num_beams, max_length=max_gen_len)
            generated = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

            f1 += calc_f1_score(generated, gts)

    return np.round(np.mean(f1) * 100, 2)
