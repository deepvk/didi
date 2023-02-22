import argparse

from torch.utils.data import DataLoader

from src.data.dataset import ConvAI2Dataset
from src.diffusion.model import DiDi
from src.diffusion.model import get_components


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
    parser.add_argument("-sc", "--schedule", default="linear", help="Noise schedule for diffusion diffusion")
    parser.add_argument("-df", "--diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("-s", "--num_steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("-l", "--logging_step", type=int, default=100, help="Logging step")
    parser.add_argument(
        "-vi", "--val_interval", type=int, default=10000, help="Number of training steps between evaluations"
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
    schedule,
    diffusion_steps,
    num_steps,
    logging_step,
    val_interval,
):
    train_dataset = ConvAI2Dataset(train, name, max_context, max_gen, have_candidates=False)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=lambda x: train_dataset.collate_fn(x, False)
    )

    val_dataset = ConvAI2Dataset(val, name, max_context, max_gen, have_candidates=False)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=lambda x: val_dataset.collate_fn(x, False)
    )

    encoder, decoder, emb_dim = get_components(name)

    model = DiDi(encoder, decoder, emb_dim, train_dataset.vocab_size, diffusion_steps, schedule)

    train(model, train_dataloader, val_dataloader, device, num_devices, num_steps, logging_step, val_interval)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
