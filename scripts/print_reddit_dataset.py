from argparse import ArgumentParser

from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import trange

from src.data.reddit_dataset import RedditDataset


def main(file_glob, tokenizer_name, seq_len, bs, skip, infinite):
    dataset = RedditDataset(file_glob, tokenizer_name, seq_len, infinite=infinite)
    dataloader = DataLoader(dataset, batch_size=bs, collate_fn=dataset.collate_fn)
    batch_iter = iter(dataloader)

    logger.info(f"Skipping {skip} samples")
    for _ in trange(skip):
        next(batch_iter)

    b_context, b_reply = next(batch_iter)
    for context, reply in zip(b_context, b_reply):
        str_context = dataset.context_tokenizer.decode(context, skip_special_tokens=False).strip()
        print(" ".join([str(it.item()) for it in context]))
        print(str_context)
        print("-" * 120)
        str_reply = dataset.reply_tokenizer.decode(reply, skip_special_tokens=False).strip()
        print(" ".join([str(it.item()) for it in reply]))
        print(str_reply)
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
