import argparse

from datasets import load_dataset

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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

        dataset = {"context": [], "candidates": [], "gt": []}

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
                        dataset["candidates"].append(candidates)
                        dataset["gt"].append(utterance1)

                    context += "<s>" + utterance1 + "</s>" + "<s>" + utterance2 + "</s>"

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["context"])

    def __getitem__(self, idx):
        sample = {
            "context": self.dataset["context"][idx],
            "candidates": self.dataset["candidates"][idx],
            "gt": self.dataset["gt"][idx],
        }

        return sample


def calc_metrics(generated_batch, candidates_batch, gt_batch, history):
    for i, candidates in enumerate(zip(*candidates_batch)):
        generated = set(generated_batch[i].split())

        f1s = []
        candidates = (gt_batch[i], *candidates)

        for candidate in candidates:
            candidate = set(candidate.split())

            precision = len(generated & candidate) / len(generated)
            recall = len(generated & candidate) / len(candidate)

            if precision + recall:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0

            f1s.append(f1)

        history["hit"].append(int(np.array(f1s).argmax() == 0))
        history["f1"].append(f1s[0])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model")
    parser.add_argument("-p", "--path")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-s", "--side", default="left")
    parser.add_argument("-l", "--max_context_len", type=int, default=128)

    args = parser.parse_args()

    dataset = ConvAI2Dataset(args.path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model, truncation_side=args.side, padding_side=args.side)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = {"f1": [], "hit": []}

    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for batch in (pbar := tqdm(dataloader)):
            pbar.set_description("Calculating metrics")
            context = batch["context"]
            candidates = batch["candidates"]
            gt = batch["gt"]

            inputs = tokenizer(context, return_tensors="pt", max_length=args.max_context_len, truncation=True, padding=True).to(device)
            reply_ids = model.generate(**inputs, max_length=64)
            output = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
            calc_metrics(output, candidates, gt, history)

    print(
        f"F1-score: {np.array(history['f1']).mean()}\nHit@1: { np.array(history['hit']).mean()}"
    )


if __name__ == "__main__":
    main()
