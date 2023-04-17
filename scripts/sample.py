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
    return parser


def main(config_path: str, model_path: str):
    config = OmegaConf.load(config_path)

    context_tokenizer = AutoTokenizer.from_pretrained(config.base_name, truncation_side="left")

    eos = context_tokenizer.eos_token
    bos = context_tokenizer.bos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_pretrained_model(model_path, config, context_tokenizer)
    model = model.to(device)

    context = ""

    while True:
        try:
            context += input("You: ")
            reply = sample(context, model, context_tokenizer, config.dataset.max_target_len, device)
            context += f"{eos}{bos}{reply}{eos}{bos}"
            print("DiDi:", reply)
        except KeyboardInterrupt:
            print("\nDiDi: Get back soon!")
            break


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
