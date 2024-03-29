import json
from typing import Iterator

import pandas as pd
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.data.utils import Preprocessor
from src.utils import zero_rank_info


class DialogSumDataset(IterableDataset):
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

        self.file = file
        self.data = pd.read_csv(file)
        self.infinite = infinite
        zero_rank_info(f"Using file: {self.file}, infinite mode: {self.infinite}")

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
            for _, row in self.data.iterrows():
                yield self.preprocess(row["dialogue"], row["summary"])
            if not self.infinite:
                break
            n_epochs += 1

    def collate_fn(self, samples: list[tuple[str, str]]):
        str_contexts, str_replies = zip(*samples)
        # [batch size; context seq len]
        contexts = self.context_tokenizer(str_contexts, max_length=self.max_context_len, **self.tokenizer_kwargs)
        # [batch size; target seq len]
        replies = self.reply_tokenizer(str_replies, max_length=self.max_target_len, **self.tokenizer_kwargs)
        return contexts, replies
