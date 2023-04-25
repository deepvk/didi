"""Script to evaluate a Seq2Seq model on ConvAI2 dataset.

As model, it is possible to use any Seq2SeqLM model from the huggingface library.
For example:
- facebook/blenderbot-400M-distill
- facebook/blenderbot_small-90M

Usage example:
python -m scripts.evaluate_model $MODEL_NAME $DATA_DIR/valid_self_revised.txt [options]

See #configure_arg_parser for more details on the options.
"""

import argparse
from os.path import join

import torch.cuda
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.reddit_dataset import RedditDataset
from src.diffusion.metrics import calculate_ppl
from src.sampling import get_pretrained_model


def configure_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("model_path", type=str, help="Path to trained model file")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")

    return parser


def main(config_path, model_path, dataset_dir):
    config = OmegaConf.load(config_path)

    context_tokenizer = AutoTokenizer.from_pretrained(config.base_name, truncation_side="left")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_pretrained_model(model_path, config, context_tokenizer)
    model = model.to(device)

    val_files_glob = join(dataset_dir, "val", "val.jsonl")
    val_dataset = RedditDataset(val_files_glob, config.base_name, infinite=False, **config.dataset)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.val_batch_size, collate_fn=val_dataset.collate_fn, pin_memory=True
    )

    ppl = calculate_ppl(model, val_dataloader, context_tokenizer.pad_token_id, config, device)

    print(f"Perplexity: {round(ppl, 2)}")


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
