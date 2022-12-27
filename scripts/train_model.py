import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import ConvAI2Dataset
from src.diffusion.model import Seq2SeqDiffusionTransformer
from src.diffusion.train_utils import collate_gt
from src.diffusion.train_utils import train_model


def configure_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help="Base transformer model name from huggingface")
    parser.add_argument("-t", "--train", help="Path to dataset for training")
    parser.add_argument("-v", "--val", help="Path to dataset for validation")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-s", "--side", default="right", help="Side of truncation and padding for tokenizer")
    parser.add_argument("-cl", "--max_context", type=int, default=256, help="Max length of context fed into diffusion")
    parser.add_argument("-gl", "--max_gen", type=int, default=64, help="Max length of generated output")
    parser.add_argument("-d", "--device", default="cpu", help="Device on which to evaluate the diffusion")
    parser.add_argument("-sc", "--schedule", default="linear", help="Noise schedule for diffusion diffusion")
    parser.add_argument("-df", "--diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("-s", "--train_steps", type=int, default=1000, help="Number of training steps")

    return parser


def main(model, train, val, batch_size, side, max_context, max_gen, device, schedule, diffusion_steps, train_steps):
    tokenizer = AutoTokenizer.from_pretrained(model, truncation_side=side, padding_side=side)

    train_dataset = ConvAI2Dataset(train, tokenizer)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=lambda x: collate_gt(x, tokenizer, max_context, max_gen)
    )

    val_dataset = ConvAI2Dataset(val)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=lambda x: collate_gt(x, tokenizer, max_context, max_gen)
    )

    model = Seq2SeqDiffusionTransformer(model, tokenizer.vocab_size, diffusion_steps).to(device)
    train_model(
        model, train_dataloader, val_dataloader, schedule, diffusion_steps, train_steps, max_context, max_gen, device
    )


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
