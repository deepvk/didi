import glob
import json
from itertools import cycle
from typing import Iterator, Iterable

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
        multiple_samples_from_threads: bool = True,
        single_turn: bool = False,
    ):
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

        self.files: Iterable[str] = glob.glob(file_glob)
        zero_rank_info(f"Using files: {', '.join(self.files)}")

        if infinite:
            zero_rank_info(f"Infinite mode is enabled, cycle files")
            self.files = cycle(self.files)

        self.multiple_samples = multiple_samples_from_threads
        self.single_turn = single_turn
        zero_rank_info(
            f"Generate multiple samples per thread: {self.multiple_samples}, use single turn context: {self.single_turn}"
        )

    @property
    def vocab_size(self) -> int:
        return self.context_tokenizer.vocab_size

    @property
    def pad_idx(self) -> int:
        return self.context_tokenizer.pad_token_id

    def __iter__(self) -> Iterator[tuple[str, str]]:
        for file in self.files:
            with open(file, "rt") as f_in:
                for line in f_in:
                    sample = json.loads(line)
                    utterances = [self.bos_token + it + self.eos_token for it in sample["thread"]]

                    if not self.multiple_samples:  # yield only one sample per thread
                        if self.single_turn:  # yield only 1 utterance per context (use 0th)
                            yield utterances[0], utterances[1]
                            continue
                        # yield full thread
                        yield " ".join(utterances[:-1]), utterances[-1]
                        continue

                    for i in range(1, len(utterances)):
                        if self.single_turn:  # yield previous utterance as context
                            yield utterances[-1], utterances[i]

                        # yield full previous thread
                        yield " ".join(utterances[:i]), utterances[i]

    def collate_fn(self, samples: list[tuple[str, str]]):
        str_contexts, str_replies = zip(*samples)
        # [batch size; context seq len]
        contexts = self.context_tokenizer(str_contexts, max_length=self.max_context_len, **self.tokenizer_kwargs)
        # [batch size; target seq len]
        replies = self.reply_tokenizer(str_replies, max_length=self.max_target_len, **self.tokenizer_kwargs)
        return contexts, replies
