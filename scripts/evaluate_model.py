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

from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from src.data.convai2_dataset import ConvAI2Dataset
from src.metrics import calculate_f1, calculate_hits_ppl


def configure_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", help="Seq2Seq model name from 🤗")
    parser.add_argument("path", help="Path to the file with the evaluation dataset")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-ml", "--max_length", type=int, default=256, help="Max length of sequences")
    parser.add_argument("-d", "--device", default="cpu", help="Device on which run model")
    parser.add_argument("-ds", "--do_sample", default=False, help="parameter for generation method")
    parser.add_argument("-nb", "--num_beams", default=5, help="Parameter for generation method")

    return parser


def main(model, path, batch_size, max_length, device, do_sample, num_beams):
    logger.info(f"Loading model and tokenizer from '{model}'")
    tokenizer = AutoTokenizer.from_pretrained(model, truncation_side="left", max_length=max_length)
    model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

    dataset = ConvAI2Dataset(path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

    hits, ppl = calculate_hits_ppl(model, dataloader, tokenizer.pad_token_id, device)
    f1 = calculate_f1(model, tokenizer, dataloader, max_length, do_sample, num_beams, device)

    print(f"Perplexity: {round(ppl, 2)}, Hits@1: {round(hits, 2)}, F1-score: {round(f1, 2)}")


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
