import argparse

from datasets import load_dataset

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.metrics import bleu_score

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ConvAI2CandidatesDataset(Dataset):
    """ConvAI2 Candidates Dataset"""

    def __init__(self, dataset):
        """
        Args:
            dataset (DatasetDict): ConvAI2 dataset from huggingface
        """

        np.random.seed(42)
        candidates_dataset = {"context": [], "candidates": []}
        ids = list(range(len(dataset["train"])))

        for i, dialog in (pbar := tqdm(enumerate(dataset["train"]))):
            pbar.set_description("Creating dataset")
            context = ""

            for utterance in dialog["dialog"]:
                if utterance["sender_class"] == "Bot" and context:
                    candidates_dataset["context"].append(context)
                    candidates = [utterance["text"]]
                    dialog_ids = [i]

                    while i in dialog_ids:
                        dialog_ids = np.random.choice(ids, size=19, replace=False)

                    for id in dialog_ids:
                        other_dialog = dataset["train"][int(id)]["dialog"]
                        length = len(other_dialog)
                        text = ""
                        sender = "Human"

                        while sender == "Human":
                            j = np.random.randint(length)
                            text = other_dialog[j]["text"]
                            sender = other_dialog[j]["sender"]
                        candidates.append(text)

                    candidates_dataset["candidates"].append(candidates)

                context += "<s>" + utterance["text"] + "</s>"
        self.dataset = candidates_dataset

    def __len__(self):
        return len(self.dataset["context"])

    def __getitem__(self, idx):
        sample = {
            "context": self.dataset["context"][idx],
            "candidates": self.dataset["candidates"][idx],
        }

        return sample


def calc_metrics(generated_batch, candidates_batch, history):
    for i, candidates in enumerate(zip(*candidates_batch)):
        bleus = []
        generated = list(generated_batch[i].split())

        for candidate in candidates:
            f1s.append(bleu_score([generated], [candidate], max_n=1, weights=[1]))

        history["hit"].append(int(np.array(bleus).argmax() == 0))
        history["bleu"].append(bleus[0])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model")
    parser.add_argument("-b", "--batch_size", type=int, default=64)

    args = parser.parse_args()

    convai2 = load_dataset("conv_ai_2")
    dataset = ConvAI2CandidatesDataset(convai2)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model, truncation_side="left", padding_side="left")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    device = "cuda" if torch.cuda_is_available() else "cpu"

    history = {"bleu": [], "hit": []}

    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for batch in (pbar := tqdm(dataloader)):
            pbar.set_description("Calculating metrics")
            context = batch["context"]
            candidates = batch["candidates"]

            inputs = tokenizer(
                context,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True,
            ).to(device)
            reply_ids = model.generate(**inputs, max_length=64)
            output = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
            calc_metrics(output, candidates, history)

    print(f"BLEU score: {np.array(history['bleu']).mean()}\nHit@1: { np.array(history['hit']).mean()}")


if __name__ == "__main__":
    main()
