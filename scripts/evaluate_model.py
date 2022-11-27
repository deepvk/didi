from evaluation_utils import ConvAI2Dataset, calculate_f1, calculate_hits_ppl

import argparse

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help="Seq2seq model name from huggingface")
    parser.add_argument("-p", "--path", help="Path to dataset for evaluation")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("-s", "--side", default="left", help="Side of truncation and padding for tokenizer")
    parser.add_argument("-cl", "--max_context", type=int, default=256, help="Max length of context fed into model")
    parser.add_argument("-gl", "--max_gen", type=int, default=64, help="Max length of generated output")
    parser.add_argument("-d", "--device", default="cpu", help="Device on which to evaluate the model")
    parser.add_argument("-ds", "--do_sample", default=True, help="parameter for generation method")
    parser.add_argument("-nb", "--num_beams", default=4, help="Parameter for generation method")

    args = parser.parse_args()

    device = args.device

    dataset = ConvAI2Dataset(args.path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model, truncation_side=args.side, padding_side=args.side)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    hits, ppl = calculate_hits_ppl(model, tokenizer, dataset, args.max_context, device)
    f1 = calculate_f1(
        model, tokenizer, dataloader, args.max_context, args.max_gen, args.do_sample, args.num_beams, device
    )

    print(f"Perplexity: {ppl}, Hits@1: {hits}, F1-score: {f1}")


if __name__ == "__main__":
    main()
