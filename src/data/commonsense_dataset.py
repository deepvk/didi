import json
from random import shuffle
from typing import Iterator

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.data.utils import Preprocessor
from src.utils import zero_rank_info


class CommonSenseDataset(IterableDataset):
    def __init__(
        self,
        file: str,
        tokenizer_name: str,
        max_context_len: int,
        infinite: bool = False,
        max_target_len: int = None,
    ):
        self.context_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, truncation_side="left"
        )
        self.reply_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, truncation_side="right"
        )
        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        self.preprocess = Preprocessor(tokenizer_name)

        self.max_context_len = max_context_len
        self.max_target_len = max_target_len or max_context_len

        self.samples = []
        with open(file, "rt") as f_in:
            for line in f_in:
                sample = json.loads(line)
                self.samples.append(self.preprocess(sample["src"], sample["trg"]))
        self.infinite = infinite
        zero_rank_info(f"Found: {len(self.samples)} samples, infinite mode: {self.infinite}")

    @property
    def vocab_size(self) -> int:
        return self.context_tokenizer.vocab_size

    @property
    def pad_idx(self) -> int:
        return self.context_tokenizer.pad_token_id

    def __iter__(self) -> Iterator[tuple[str, str]]:
        n_epochs = 0
        while True:
            zero_rank_info(f"Start epoch {n_epochs}")
            yield from self.samples
            if not self.infinite:
                break
            n_epochs += 1
            shuffle(self.samples)

    def collate_fn(self, samples: list[tuple[str, str]]):
        str_contexts, str_replies = zip(*samples)
        # [batch size; context seq len]
        contexts = self.context_tokenizer(str_contexts, max_length=self.max_context_len, **self.tokenizer_kwargs)
        # [batch size; target seq len]
        replies = self.reply_tokenizer(str_replies, max_length=self.max_target_len, **self.tokenizer_kwargs)
        return contexts, replies

    def test_collate_fn(self, samples: list[tuple[str, str]]):
        str_contexts, str_replies = zip(*samples)
        # [batch size; context seq len]
        contexts = self.context_tokenizer(str_contexts, max_length=self.max_context_len, **self.tokenizer_kwargs)
        return contexts, str_contexts, str_replies
