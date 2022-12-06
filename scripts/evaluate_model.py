from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from evaluation_utils import ConvAI2Dataset
from evaluation_utils import calculate_f1
from evaluation_utils import calculate_hits_ppl
from evaluation_utils import configure_arg_parser


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
