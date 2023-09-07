import argparse
import json
from os.path import join

import torch.cuda
from lightning import seed_everything
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.commonsense_dataset import CommonSenseDataset
from src.data.utils import Preprocessor
from src.diffusion.model import DiDi, get_components
from src.sampling import sample


def configure_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("model_path", type=str, help="Path to trained model file")
    parser.add_argument("seeds", nargs="*", type=int, help="Random seeds")
    parser.add_argument("-m", "--mode", type=str, default="ddpm", help="Sampling mode")
    parser.add_argument("-f", "--freq", type=int, default=1, help="Sampling step frequency")
    parser.add_argument("-i", "--dataset_dir", type=str, help="Input file for generation")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory for sampling result")
    parser.add_argument("-d", "--device_id", type=int, default=0, help="GPU device id")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    return parser


def main(
    config_path: str,
    model_path: str,
    seeds: list[int],
    mode: str,
    freq: int,
    dataset_dir: str,
    output_dir: str,
    device_id: int,
    batch_size: int,
):
    config = OmegaConf.load(config_path)

    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
        "add_special_tokens": False,
    }

    preprocess = Preprocessor(config.base_name)
    context_tokenizer = AutoTokenizer.from_pretrained(config.base_name, truncation_side="left")

    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    _, _, enc_dim, dec_dim = get_components(config.base_name)
    model = DiDi.load_from_checkpoint(model_path, enc_dim=enc_dim, dec_dim=dec_dim, map_location=device)
    model.eval()

    if dataset_dir is None:
        context = []
        while True:
            try:
                utterance = input("You: ")
                context.append(utterance)
                joined_context, _ = preprocess(context, "")
                raw_context = context_tokenizer(joined_context, **tokenizer_kwargs).to(device)
                reply = sample(raw_context, model, mode, freq, context_tokenizer)[0]
                context.append(reply)
                print("DiDi:", reply)
            except KeyboardInterrupt:
                print("\nDiDi: Get back soon!")
                break
        return

    test_files_glob = join(dataset_dir, "test.jsonl")
    test_dataset = CommonSenseDataset(test_files_glob, config.base_name, **config.dataset)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.test_collate_fn,
        pin_memory=True,
        num_workers=1,
    )

    for seed in seeds:
        seed_everything(seed)

        with open(join(output_dir, f"seed{seed}_{mode}_freq{freq}.json"), "w") as f:
            for batch in tqdm(test_dataloader):
                raw_context, context, target = batch
                raw_context = raw_context.to(device)
                with torch.inference_mode():
                    predictions = sample(
                        raw_context, model, mode, freq, tokenizer=test_dataset.reply_tokenizer, skip_special=False
                    )

                for recov, ref, src in zip(predictions, target, context):
                    print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=f)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
