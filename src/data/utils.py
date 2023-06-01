from transformers import AutoTokenizer

from src.diffusion.model import get_mode, Modes


class Preprocessor:
    def __init__(self, base_name):
        mode = get_mode(base_name)
        tokenizer = AutoTokenizer.from_pretrained(base_name)

        if mode is Modes.BERT:
            self.bos, self.sep, self.eos = "", tokenizer.sep_token, tokenizer.sep_token
        elif mode is Modes.BLENDERBOT:
            self.bos, self.eos = tokenizer.bos_token, tokenizer.eos_token
            self.sep = f"{self.bos} {self.eos}"
        else:
            self.bos, self.sep, self.eos = "", "\n", tokenizer.eos_token

    def __call__(self, src, trg):
        if isinstance(src, list):
            src = self.sep.join(src)
        return self.bos + src + self.eos, self.bos + trg + self.eos
