import argparse
from os.path import join

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.reddit_dataset import RedditDataset
from src.diffusion.model import DiDi
from src.diffusion.model import get_components
from src.training import train_model
from src.utils import filter_warnings, setup_logger, zero_rank_info


def configure_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    return parser


def main(config_path: str, dataset_dir: str):
    filter_warnings()
    setup_logger()
    torch.set_float32_matmul_precision("high")

    config = OmegaConf.load(config_path)
    zero_rank_info(f"Loaded config:\n{OmegaConf.to_yaml(config, resolve=False, sort_keys=True)}")

    train_files_glob = join(dataset_dir, "train", "*.jsonl.gz")
    train_dataset = RedditDataset(train_files_glob, config.base_name, infinite=True, **config.dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    val_files_glob = join(dataset_dir, "val", "*.jsonl.gz")
    val_dataset = RedditDataset(val_files_glob, config.base_name, infinite=False, **config.dataset)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, collate_fn=val_dataset.collate_fn, pin_memory=True
    )

    encoder, decoder, emb_dim = get_components(config.base_name, **config.decoder)
    model = DiDi(encoder, decoder, emb_dim, train_dataset.vocab_size, **config.didi)
    # model = torch.compile(model)

    train_model(
        model, train_dataloader, val_dataloader, config.trainer, seed=config.seed, save_interval=config.save_interval
    )


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
