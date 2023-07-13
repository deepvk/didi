import argparse
from os import environ
from os.path import join

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.conditioning.adapter import Adapter
from src.data.convai2_dataset import ConvAI2Dataset
from src.diffusion.model import DiDi
from src.pipeline.training import train_model
from src.utils import filter_warnings, setup_logger, zero_rank_info


def configure_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("model_path", type=str, help="Path to DiDi model")
    parser.add_argument("--condition", type=str, default="other", help="Type of persona")
    return parser


def main(config_path: str, dataset_dir: str, model_path: str, condition: str):
    filter_warnings()
    setup_logger()
    environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.set_float32_matmul_precision("high")

    config = OmegaConf.load(config_path)
    zero_rank_info(f"Loaded config:\n{OmegaConf.to_yaml(config, resolve=False, sort_keys=True)}")

    train_dataset = ConvAI2Dataset(
        join(dataset_dir, f"train_{condition}_revised_no_cands.txt"), config.base_name, **config.dataset
    )
    val_dataset = ConvAI2Dataset(
        join(dataset_dir, f"valid_{condition}_revised_no_cands.txt"), config.base_name, **config.dataset
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        num_workers=1,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        num_workers=1,
    )

    didi = DiDi.load_from_checkpoint(model_path)
    model = Adapter(didi)

    train_model(
        model,
        train_dataloader,
        val_dataloader,
        config.trainer,
        seed=config.seed,
        save_interval=config.save_interval,
    )


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
