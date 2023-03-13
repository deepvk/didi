import json
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


@dataclass
class Dialog:
    context: list[str]
    candidates: list[str]  # first candidate is the correct one, the rest are distractions
    my_persona: Optional[list[str]] = None
    partner_persona: Optional[list[str]] = None


class ConvAI2Dataset(Dataset):
    _YOUR_PERSONA_PREFIX = "your persona: "
    _PARTNER_PERSONA_PREFIX = "partner's persona: "

    def __init__(self, path, tokenizer_name, max_context_len, max_target_len=None, have_candidates=True):
        self.dataset = []
        self.num_dialogs = 0

        self.context_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, truncation_side="left")
        self.candidate_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        bos = self.context_tokenizer.bos_token
        eos = self.context_tokenizer.eos_token

        self.max_context_len = max_context_len
        self.max_target_len = max_target_len or max_context_len

        self.have_candidates = have_candidates
        self.vocab_size = self.context_tokenizer.vocab_size

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

                if have_candidates:
                    utterance1, utterance2, _, candidates_str = line.split("\t")
                else:
                    utterance1, utterance2, *_ = line.split("\t")
                    candidates_str = ""

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

    def collate_fn(self, samples: list[Dialog], return_all_candidates: bool = False):
        return_all_candidates = self.have_candidates & return_all_candidates
        str_contexts = [" ".join(sample.context) for sample in samples]
        # [batch size, context seq len]
        b_contexts = self.context_tokenizer(
            str_contexts,
            max_length=self.max_context_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        if return_all_candidates:
            str_candidates = [it for sample in samples for it in sample.candidates]
        else:
            str_candidates = [sample.candidates[0] for sample in samples]

        # Tokenizer truncates on the left, but for candidates we want to truncate on the right
        b_candidates = self.candidate_tokenizer(
            str_candidates, padding="max_length", return_tensors="pt", add_special_tokens=False
        ).input_ids
        b_candidates = b_candidates[:, : self.max_target_len]
        # [batch size, # candidates, candidates seq len]
        b_candidates = b_candidates.view(len(samples), -1, b_candidates.size(1))

        return b_contexts, b_candidates.squeeze(1)


class CommonsenseConversationDataset(Dataset):
    def __init__(self, path, tokenizer_name, max_context_len, max_target_len=None):
        self.dataset = []

        self.context_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, truncation_side="left")
        self.reply_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.max_context_len = max_context_len
        self.max_target_len = max_target_len or max_context_len

        self.vocab_size = self.context_tokenizer.vocab_size

        logger.info(f"Loading dataset from '{path}'")

        with open(path, "r") as f:
            lines_list = list(f)

        for line in lines_list:
            result = json.loads(line)
            self.dataset.append((result["src"], result["trg"]))

        logger.info(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.dataset[idx]

    def collate_fn(self, samples: list[tuple[str, str]]):
        str_contexts = [context for context, _ in samples]
        # [batch size, context seq len]
        b_contexts = self.context_tokenizer(
            str_contexts,
            max_length=self.max_context_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        str_replies = [reply for _, reply in samples]

        # Tokenizer truncates on the left, but for replies we want to truncate on the right
        b_replies = self.reply_tokenizer(
            str_replies, padding="max_length", return_tensors="pt", add_special_tokens=False
        ).input_ids
        b_replies = b_replies[:, : self.max_target_len]

        return b_contexts, b_replies
