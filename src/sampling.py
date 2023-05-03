import torch

from src.diffusion.model import DiDi
from src.diffusion.model import get_components


def get_pretrained_model(model_path, config, tokenizer):
    encoder, decoder, emb_dim = get_components(config.base_name, **config.decoder)
    model = DiDi(encoder, decoder, emb_dim, tokenizer.vocab_size, pad_idx=tokenizer.pad_token_id, **config.didi)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return model


@torch.no_grad()
def sample(raw_context, model, tokenizer, mode, step_freq, device, raw_output=False):
    input_ids = raw_context.input_ids
    emb = model.emb(input_ids)

    x_t = torch.randn_like(emb) * model.sigmas[-1]

    cached_context = None
    ones = torch.ones((emb.shape[0], 1), dtype=torch.long, device=device)
    noise = torch.empty_like(emb)

    if mode == "ddpm":
        logits = model.sample_ddpm(model, x_t, raw_context, cached_context, noise, ones, step_freq)
    elif mode == "euler":
        logits = model.sample_euler(model, x_t, raw_context, cached_context, noise, ones, step_freq)
    else:
        raise NotImplementedError(f"No {mode} sampling strategy")

    predictions = logits.argmax(-1)
    if raw_output:
        return predictions

    eos_id = tokenizer.eos_token_id
    replies = tokenizer.batch_decode(truncate_predictions(predictions, eos_id), skip_special_tokens=True)
    return select_reply(replies)


def truncate_predictions(predictions, eos_id):
    res = []

    for prediction in predictions:
        idx = (prediction == eos_id).nonzero()
        if len(idx):
            res.append(prediction[: idx[0] + 1])
        else:
            res.append(prediction)
    return res


def select_reply(replies):
    return replies
