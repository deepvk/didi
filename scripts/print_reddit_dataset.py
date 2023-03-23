from argparse import ArgumentParser

from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import trange

from src.data.reddit_dataset import RedditDataset


def print_sample(batch, idx, tokenizer):
    print(" ".join([str(it.item()) for it in batch.attention_mask[idx]]))
    print(" ".join([str(it.item()) for it in batch.input_ids[idx]]))
    print(tokenizer.decode(batch.input_ids[idx]))


def main(file_glob, tokenizer_name, seq_len, bs, skip, infinite):
    dataset = RedditDataset(file_glob, tokenizer_name, seq_len, infinite=infinite)
    dataloader = DataLoader(dataset, batch_size=bs, collate_fn=dataset.collate_fn)
    batch_iter = iter(dataloader)

    logger.info(f"Skipping {skip} samples")
    for _ in trange(skip):
        next(batch_iter)

    b_context, b_reply = next(batch_iter)
    for i in range(bs):
        print_sample(b_context, i, dataset.context_tokenizer)
        print("-" * 120)
        print_sample(b_reply, i, dataset.reply_tokenizer)
        print("=" * 120)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("file_glob", type=str, help="Glob with dataset files")
    arg_parser.add_argument("tokenizer_name", type=str, help="Tokenizer name")
    arg_parser.add_argument("--seq-len", type=int, default=32, help="Sequence length to use")
    arg_parser.add_argument("--bs", type=int, default=3, help="Batch size")
    arg_parser.add_argument("--skip", type=int, default=0, help="Number of batches to skip")
    arg_parser.add_argument("--infinite", action="store_true", help="If passed, enable infinite mode for dataset")
    main(**vars(arg_parser.parse_args()))
