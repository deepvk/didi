import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import ConvAI2Dataset
from src.diffusion.model import DiffusionTransformer
from src.diffusion.train_utils import train_model


def configure_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--diffusion", help="Base diffusion name from huggingface")
    parser.add_argument("-p", "--path", help="Path to dataset for training")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-s", "--side", default="left", help="Side of truncation and padding for tokenizer")
    parser.add_argument("-cl", "--max_context", type=int, default=256, help="Max length of context fed into diffusion")
    parser.add_argument("-gl", "--max_gen", type=int, default=64, help="Max length of generated output")
    parser.add_argument("-d", "--device", default="cpu", help="Device on which to evaluate the diffusion")
    parser.add_argument("-sc", "--schedule", default="linear", help="Noise schedule for diffusion diffusion")
    parser.add_argument("-df", "--diffusion_steps", type=int, default=5, help="Number of diffusion steps")
    parser.add_argument("-e", "--num_epochs", type=int, default=1, help="Number of training epochs")

    return parser


def main(model, path, batch_size, side, max_context, max_gen, device, schedule, diffusion_steps, num_epochs):
    dataset = ConvAI2Dataset(path)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    tokenizer = AutoTokenizer.from_pretrained(model, truncation_side=side, padding_side=side)

    model = DiffusionTransformer(model, diffusion_steps).to(device)
    train_model(model, tokenizer, dataloader, schedule, diffusion_steps, num_epochs, max_context, max_gen, device)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
