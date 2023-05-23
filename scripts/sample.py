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
from src.diffusion.model import DiDi
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
):
    config = OmegaConf.load(config_path)

    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
        "add_special_tokens": False,
    }

    context_tokenizer = AutoTokenizer.from_pretrained(config.base_name, truncation_side="left")
    bos = context_tokenizer.bos_token
    eos = context_tokenizer.eos_token

    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    model = DiDi.load_from_checkpoint(model_path).to(device)
    model.eval()

    if dataset_dir is None:
        context = f"{bos} "
        while True:
            try:
                context += f"{input('You: ')}{eos}"
                raw_context = context_tokenizer(context, **tokenizer_kwargs).to(device)
                reply = sample(raw_context, model, mode, freq, context_tokenizer)[0]
                context += f"{eos} {bos} {reply}{eos} {bos}"
                print("DiDi:", reply)
            except KeyboardInterrupt:
                print("\nDiDi: Get back soon!")
                break
    else:
        test_files_glob = join(dataset_dir, "test.jsonl")
        test_dataset = CommonSenseDataset(test_files_glob, config.base_name, **config.dataset)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.val_batch_size,
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
                    predictions = sample(
                        raw_context, model, mode, freq, test_dataset.reply_tokenizer, skip_special=False
                    )

                    for recov, ref, src in zip(predictions, target, context):
                        print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=f)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))
