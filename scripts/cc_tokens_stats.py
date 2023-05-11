import json
from argparse import ArgumentParser
from collections import Counter
from os.path import join, basename

from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str, help="Path to CommonSense dataset")
    arg_parser.add_argument("tokenizer", type=str, help="Name of the tokenizer")
    return arg_parser


def collect_stats(filepath: str, tokenizer: PreTrainedTokenizer):
    src_counter, trg_counter = Counter(), Counter()
    with open(filepath, "r") as f_in:
        for line in tqdm(f_in, desc=basename(filepath), unit=" lines", unit_scale=True):
            sample = json.loads(line)
            src, trg = sample["src"], sample["trg"]
            tokens = tokenizer.batch_encode_plus([src, trg], padding=False, truncation=False, add_special_tokens=False)
            src_counter[len(tokens[0])] += 1
            trg_counter[len(tokens[1])] += 1
    return src_counter, trg_counter


def print_counter_stats(counter: Counter, name: str):
    min_len = min(counter.keys())
    max_len = max(counter.keys())
    total = sum([k * v for k, v in counter.items()])
    average = total / sum(counter.values())
    logger.info(f"[{name}] Total tokens: {total}, range: {min_len} - {max_len}, average: {average:.3f}")


def main(data: str, tokenizer: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    logger.info(f"Tokenizer: {tokenizer}")

    for split in ["train", "valid", "test"]:
        src_counter, trg_counter = collect_stats(join(data, f"{split}.jsonl"), tokenizer)
        print_counter_stats(src_counter, "src")
        print_counter_stats(trg_counter, "trg")


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
