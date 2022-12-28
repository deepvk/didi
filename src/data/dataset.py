from dataclasses import dataclass

from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm


@dataclass
class Dialog:
    context: list[str]
    candidates: list[str]  # first candidate is the correct one, the rest are distractions
    my_persona: list[str]
    partner_persona: list[str]


class ConvAI2Dataset(Dataset):
    _YOUR_PERSONA_PREFIX = "your persona: "
    _PARTNER_PERSONA_PREFIX = "partner's persona: "

    def __init__(self, path, tokenizer, max_seq_len):
        self.dataset = []
        self.num_dialogs = 0

        bos = tokenizer.bos_token
        eos = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            "add_special_tokens": False,
            "max_length": max_seq_len,
        }

        logger.info(f"Loading dataset from '{path}'")
        with open(path, "r") as f:
            for line in tqdm(f, desc="Loading dataset"):
                line_no, line = line.strip().split(" ", 1)
                line_no = int(line_no)

                # New dialog
                if line_no == 1:
                    self.num_dialogs += 1
                    context, my_persona, partner_persona = [], [], []

                if line.startswith(self._YOUR_PERSONA_PREFIX):
                    my_persona.append(line[len(self._YOUR_PERSONA_PREFIX) :])
                    continue
                if line.startswith(self._PARTNER_PERSONA_PREFIX):
                    partner_persona.append(line[len(self._PARTNER_PERSONA_PREFIX) :])
                    continue

                utterance1, utterance2, _, candidates_str = line.split("\t")
                utterance1 = bos + utterance1 + eos
                utterance2 = bos + utterance2 + eos
                # Last candidate is always the correct one (same as utterance2), put it first
                candidates = [utterance2] + [bos + candidate + eos for candidate in candidates_str.split("|")[:-1]]

                context.append(utterance1)
                self.dataset.append(Dialog(context, candidates, my_persona, partner_persona))
                context.append(utterance2)
        logger.info(f"Loaded {len(self.dataset)} samples from {self.num_dialogs} dialogs")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dialog:
        return self.dataset[idx]

    def collate_fn(self, samples: list[Dialog]):
        str_contexts = [" ".join(sample.context) for sample in samples]
        # [batch size, context seq len]
        b_contexts = self.tokenizer(str_contexts, **self.tokenizer_kwargs).input_ids

        str_candidates = [it for sample in samples for it in sample.candidates]
        b_candidates = self.tokenizer(str_candidates, **self.tokenizer_kwargs).input_ids
        # [batch size, # candidates, candidates seq len]
        b_candidates = b_candidates.view(len(samples), -1, b_candidates.size(1))

        return b_contexts, b_candidates
