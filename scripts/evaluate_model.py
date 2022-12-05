import argparse

from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from src.data.dataset import ConvAI2Dataset
from src.metrics import calculate_f1
from src.metrics import calculate_hits_ppl


def configure_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--diffusion", help="Seq2seq diffusion name from huggingface")
    parser.add_argument("-p", "--path", help="Path to dataset for evaluation")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("-s", "--side", default="left", help="Side of truncation and padding for tokenizer")
    parser.add_argument("-cl", "--max_context", type=int, default=256, help="Max length of context fed into diffusion")
    parser.add_argument("-gl", "--max_gen", type=int, default=64, help="Max length of generated output")
    parser.add_argument("-d", "--device", default="cpu", help="Device on which to evaluate the diffusion")
    parser.add_argument("-ds", "--do_sample", default=True, help="parameter for generation method")
    parser.add_argument("-nb", "--num_beams", default=4, help="Parameter for generation method")

    return parser


def main(model, path, batch_size, side, max_context, max_gen, device, do_sample, num_beams):
    dataset = ConvAI2Dataset(path)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    tokenizer = AutoTokenizer.from_pretrained(model, truncation_side=side, padding_side=side)
    model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

    hits, ppl = calculate_hits_ppl(model, tokenizer, dataset, max_context, device)
    f1 = calculate_f1(model, tokenizer, dataloader, max_context, max_gen, do_sample, num_beams, device)

    print(f"Perplexity: {ppl}, Hits@1: {hits}, F1-score: {f1}")


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
