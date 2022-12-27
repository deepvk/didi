from torch.utils.data import Dataset


class ConvAI2Dataset(Dataset):
    _PERSONA_PREFIX = "your persona"

    def __init__(self, path, tokenizer):
        context = ""

        candidates = []

        dataset = {"context": [], "candidates": []}

        bos = tokenizer.bos_token
        eos = tokenizer.eos_token

        with open(path, "r") as f:
            for line in f.readlines():
                if line[2:].startswith(self._PERSONA_PREFIX):
                    context = ""
                    candidates = []
                    continue

                else:
                    prev_candidates = candidates

                    utterance1, utterance2, _, candidates_str = line[2:].split("\t")
                    candidates = [bos + candidate + eos for candidate in candidates_str.split("|")[:-1]]

                    if prev_candidates:
                        dataset["context"].append(context)
                        dataset["candidates"].append([bos + utterance1 + eos] + candidates)

                        self.num_candidates = len(candidates) + 1

                    context += bos + utterance1 + eos + bos + utterance2 + eos

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["context"])

    def __getitem__(self, idx):
        sample = {
            "context": self.dataset["context"][idx],
            "candidates": self.dataset["candidates"][idx],
        }

        return sample
