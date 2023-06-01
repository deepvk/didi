import argparse
from functools import partial
from os import environ
from os.path import join

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.commonsense_dataset import CommonSenseDataset
from src.data.distributed_dataset import DistributedIterableDataset
from src.data.reddit_dataset import RedditDataset
from src.diffusion.model import DiDi
from src.diffusion.model import get_components
from src.training import train_model
from src.utils import filter_warnings, setup_logger, zero_rank_info


def configure_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--ckpt_dir", type=str, help="Path to checkpoint directory.")
    parser.add_argument("--resume", type=str, help="Path to checkpoint file to resume training.")
    parser.add_argument("--commonsense", action="store_true", help="Whether to use Reddit or CommonSense dataset.")
    return parser


def main(config_path: str, dataset_dir: str, ckpt_dir: str = None, resume: str = None, commonsense: bool = False):
    filter_warnings()
    setup_logger()
    environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.set_float32_matmul_precision("high")

    config = OmegaConf.load(config_path)
    zero_rank_info(f"Loaded config:\n{OmegaConf.to_yaml(config, resolve=False, sort_keys=True)}")

    if commonsense:
        train_dataset = CommonSenseDataset(
            join(dataset_dir, "train.jsonl"), config.base_name, infinite=True, **config.dataset
        )
        val_dataset = CommonSenseDataset(
            join(dataset_dir, "valid.jsonl"), config.base_name, infinite=False, **config.dataset
        )
    else:
        train_files_glob = join(dataset_dir, "train", "train.jsonl-*")
        train_dataset = RedditDataset(train_files_glob, config.base_name, infinite=True, **config.dataset)

        val_files_glob = join(dataset_dir, "val.jsonl")
        val_dataset = RedditDataset(val_files_glob, config.base_name, infinite=False, **config.dataset)

    train_dataloader = DataLoader(
        DistributedIterableDataset(train_dataset),
        batch_size=config.batch_size,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        num_workers=1,
    )

    val_dataloader = DataLoader(
        DistributedIterableDataset(val_dataset),
        batch_size=config.val_batch_size,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        num_workers=1,
    )

    encoder, decoder, enc_dim, dec_dim = get_components(config.base_name, **config.decoder)
    batch_decoder = partial(train_dataset.reply_tokenizer.batch_decode, skip_special_tokens=False)
    model = DiDi(
        encoder,
        decoder,
        enc_dim,
        dec_dim,
        train_dataset.vocab_size,
        pad_idx=train_dataset.pad_idx,
        batch_decoder=batch_decoder,
        **config.didi,
    )

    train_model(
        model,
        train_dataloader,
        val_dataloader,
        config.trainer,
        seed=config.seed,
        save_interval=config.save_interval,
        ckpt_dir=ckpt_dir,
        resume=resume,
    )


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
