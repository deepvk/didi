import argparse

from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from src.data.dataset import ConvAI2Dataset
from src.metrics import calculate_f1
from src.metrics import calculate_hits_ppl
from src.metrics import collate_candidates
from src.metrics import collate_gt


def configure_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--diffusion", help="Seq2seq diffusion name from huggingface")
    parser.add_argument("-p", "--path", help="Path to dataset for evaluation")
    parser.add_argument("-bp", "--batch_size_ppl", type=int, default=64, help="Batch size for hits and ppl evaluation")
    parser.add_argument("-bf", "--batch_size_f1", type=int, default=8, help="Batch size for f1 evaluation")
    parser.add_argument("-s", "--side", default="left", help="Side of truncation and padding for tokenizer")
    parser.add_argument("-cl", "--max_context", type=int, default=256, help="Max length of context fed into diffusion")
    parser.add_argument(
        "-sl", "--max_candidates", type=int, default=64, help="Max length of candidates fed into diffusion"
    )
    parser.add_argument("-gl", "--max_gen", type=int, default=64, help="Max length of generated output")
    parser.add_argument("-d", "--device", default="cpu", help="Device on which to evaluate the diffusion")
    parser.add_argument("-ds", "--do_sample", default=True, help="parameter for generation method")
    parser.add_argument("-nb", "--num_beams", default=4, help="Parameter for generation method")

    return parser


def main(
    model, path, batch_size_ppl, batch_size_f1, side, max_context, max_candidates, max_gen, device, do_sample, num_beams
):
    dataset = ConvAI2Dataset(path)

    tokenizer = AutoTokenizer.from_pretrained(model, truncation_side=side, padding_side=side)
    model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)

    dataloader_hits_ppl = DataLoader(
        dataset,
        batch_size=batch_size_ppl,
        collate_fn=lambda x: collate_candidates(x, tokenizer, max_context, max_candidates),
    )

    dataloader_f1 = DataLoader(
        dataset, batch_size=batch_size_f1, collate_fn=lambda x: collate_gt(x, tokenizer, max_context)
    )

    hits, ppl = calculate_hits_ppl(model, dataloader_hits_ppl, max_context, device)
    f1 = calculate_f1(model, tokenizer, dataloader_f1, max_gen, do_sample, num_beams, device)

    print(f"Perplexity: {ppl}, Hits@1: {hits}, F1-score: {f1}")


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
