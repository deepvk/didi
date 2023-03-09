import argparse

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CommonsenseConversationDataset
from src.data.dataset import ConvAI2Dataset
from src.diffusion.model import DiDi
from src.diffusion.model import get_components
from src.training import train_model

torch.set_float32_matmul_precision("high")


def configure_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--name", help="Base transformer model name from huggingface")
    parser.add_argument("-t", "--train", help="Path to dataset for training")
    parser.add_argument("-v", "--val", help="Path to dataset for validation")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-cl", "--max_context", type=int, default=256, help="Max length of context fed into diffusion")
    parser.add_argument("-gl", "--max_gen", type=int, default=32, help="Max length of generated output")
    parser.add_argument("-d", "--device", default="cpu", help="Device on which to evaluate the diffusion")
    parser.add_argument("-nd", "--num_devices", type=int, default=1, help="Number of devices")
    parser.add_argument("-nw", "--num_workers", type=int, default=1, help="Number of workers for dataloader")
    parser.add_argument("-sc", "--schedule", default="sqrt", help="Noise schedule for diffusion diffusion")
    parser.add_argument("-df", "--diffusion_steps", type=int, default=2000, help="Number of diffusion steps")
    parser.add_argument("-s", "--num_steps", type=int, default=100000, help="Number of training steps")
    parser.add_argument("-l", "--logging_step", type=int, default=100, help="Logging step")
    parser.add_argument(
        "-vi", "--val_interval", type=int, default=10000, help="Number of training steps between evaluations"
    )
    parser.add_argument("-p", "--pretrain", action="store_true", help="Flag for model pretraining")
    parser.add_argument("-dm", "--decoder_mode", default="bert", help="Model decoder type")
    parser.add_argument(
        "-sf", "--step_freq", type=int, default=10, help="Number of skipped diffusion steps during decoding"
    )

    return parser


def main(
    name,
    train,
    val,
    batch_size,
    max_context,
    max_gen,
    device,
    num_devices,
    num_workers,
    schedule,
    diffusion_steps,
    num_steps,
    logging_step,
    val_interval,
    pretrain,
    decoder_mode,
    step_freq,
):
    if pretrain:
        train_dataset = CommonsenseConversationDataset(train, name, max_context, max_gen)
        val_dataset = CommonsenseConversationDataset(val, name, max_context, max_gen)
    else:
        train_dataset = ConvAI2Dataset(train, name, max_context, max_gen, have_candidates=False)
        val_dataset = ConvAI2Dataset(val, name, max_context, max_gen, have_candidates=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
    )

    encoder, decoder, emb_dim = get_components(name, decoder_mode)

    model = DiDi(encoder, decoder, emb_dim, train_dataset.vocab_size, diffusion_steps, schedule, step_freq)

    train_model(model, train_dataloader, val_dataloader, device, num_devices, num_steps, logging_step, val_interval)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
