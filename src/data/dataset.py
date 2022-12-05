from torch.utils.data import Dataset


class ConvAI2Dataset(Dataset):
    _PERSONA_PREFIX = "your persona"

    def __init__(self, path):
        context = ""

        candidates = []

        dataset = {"context": [], "candidates": []}

        with open(path, "r") as f:
            for line in f.readlines():
                if line[2:].startswith(self._PERSONA_PREFIX):
                    context = ""
                    candidates = []
                    continue

                else:
                    prev_candidates = candidates

                    utterance1, utterance2, _, candidates_str = line[2:].split("\t")
                    candidates = candidates_str.split("|")[:-1]

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
