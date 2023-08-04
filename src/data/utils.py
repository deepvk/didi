from transformers import AutoTokenizer

from src.diffusion.model import get_mode, Modes
import torch


class Preprocessor:
    def __init__(self, base_name: str, context_dropout_prob: float = 0.0):
        mode = get_mode(base_name)
        tokenizer = AutoTokenizer.from_pretrained(base_name)

        self.dropout_prob = context_dropout_prob

        if mode is Modes.BERT:
            self.bos, self.eos = tokenizer.cls_token, tokenizer.sep_token
            self.sep = f"{self.bos} {self.eos}"
        elif mode is Modes.BLENDERBOT:
            self.bos, self.eos = tokenizer.bos_token, tokenizer.eos_token
            self.sep = f"{self.bos} {self.eos}"
        else:
            self.bos, self.sep, self.eos = "", "\n", tokenizer.eos_token

    def __call__(self, src, trg):
        if torch.rand(1) < self.dropout_prob:
            src = ""
        elif isinstance(src, list):
            src = self.sep.join(src)
        return self.bos + src + self.eos, self.bos + trg + self.eos, self.bos + "" + self.eos
