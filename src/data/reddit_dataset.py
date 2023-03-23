import glob
import json
from itertools import cycle
from os import environ
from typing import Iterator, Iterable

from loguru import logger
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.utils import zero_rank_info


class RedditDataset(IterableDataset):
    def __init__(
        self,
        file_glob: str,
        tokenizer_name: str,
        max_context_len: int,
        max_target_len: int = None,
        infinite: bool = False,
    ):
        self._ws, self._rank = 1, 0

        self.context_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, truncation_side="left"
        )
        self.reply_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, truncation_side="right"
        )
        self.tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }

        self.max_context_len = max_context_len
        self.max_target_len = max_target_len or max_context_len

        self.bos_token = self.context_tokenizer.bos_token
        self.eos_token = self.context_tokenizer.eos_token

        self.files = glob.glob(file_glob)
        zero_rank_info(f"Using files: {', '.join(self.files)}")

        self.infinite = infinite
        zero_rank_info(f"Infinite mode is enabled")

    @property
    def vocab_size(self) -> int:
        return self.context_tokenizer.vocab_size

    @property
    def pad_idx(self) -> int:
        return self.context_tokenizer.pad_token_id

    def _log(self, msg: str):
        if self._ws > 1:
            msg = f"[Dataset {self._rank + 1}/{self._ws}] " + msg
        logger.info(msg)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        self._ws, self._rank = int(environ.get("WORLD_SIZE", 1)), int(environ.get("LOCAL_RANK", 0))

        shards: Iterable[str] = self.files[self._rank :: self._ws]
        self._log(f"Shards to use (infinite: {self.infinite}): {', '.join(shards)}")
        if self.infinite:
            shards = cycle(shards)

        for shard in shards:
            self._log(f"Processing {shard}")
            with open(shard, "rt") as f_in:
                for line in f_in:
                    sample = json.loads(line)
                    utterances = [self.bos_token + it + self.eos_token for it in sample["thread"]]
                    for i in range(1, len(utterances)):
                        yield " ".join(utterances[:i]), utterances[i]

    def collate_fn(self, samples: list[tuple[str, str]]):
        str_contexts, str_replies = zip(*samples)
        # [batch size; context seq len]
        contexts = self.context_tokenizer(str_contexts, max_length=self.max_context_len, **self.tokenizer_kwargs)
        # [batch size; target seq len]
        replies = self.reply_tokenizer(str_replies, max_length=self.max_target_len, **self.tokenizer_kwargs)
        return contexts, replies
