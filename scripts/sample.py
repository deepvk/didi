import argparse

import torch.cuda
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.sampling import get_pretrained_model
from src.sampling import sample


def configure_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("model_path", type=str, help="Path to trained model file")
    parser.add_argument("mode", type=str, nargs="?", default="ddpm", help="Sampling mode")
    return parser


def main(config_path: str, model_path: str, mode: str):
    config = OmegaConf.load(config_path)

    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
        "add_special_tokens": False,
    }

    context_tokenizer = AutoTokenizer.from_pretrained(config.base_name, truncation_side="left")

    bos = context_tokenizer.bos_token
    eos = context_tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_pretrained_model(model_path, config, context_tokenizer)
    model = model.to(device)

    context = f"{bos} "

    while True:
        try:
            context += f"{input('You: ')}{eos}"
            raw_context = context_tokenizer(context, **tokenizer_kwargs).to(device)
            reply = sample(raw_context, model, context_tokenizer, mode, config.didi.step_freq, device)[0]
            context += f"{eos} {bos} {reply}{eos} {bos}"
            print("DiDi:", reply)
        except KeyboardInterrupt:
            print("\nDiDi: Get back soon!")
            break


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
