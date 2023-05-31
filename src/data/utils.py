from transformers import AutoTokenizer

from src.diffusion.model import get_mode, Modes


class Preprocessor:
    def __init__(self, base_name):
        mode = get_mode(base_name)
        tokenizer = AutoTokenizer.from_pretrained(base_name)

        self.bos = tokenizer.bos_token if mode is Modes.BLENDERBOT else ""
        self.eos = tokenizer.eos_token
        self.sep = f"{self.bos} {self.eos}" if mode is Modes.BLENDERBOT else "\n"

    def __call__(self, src, trg):
        if isinstance(src, list):
            src = self.sep.join(src)
        return self.bos + src + self.eos, self.bos + trg + self.eos
