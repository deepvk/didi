import torch

from src.diffusion.model import DiDi
from src.diffusion.model import get_components
from src.diffusion.utils import scale_input


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
        logits = sample_ddpm(model, x_t, raw_context, cached_context, noise, ones, step_freq)
    elif mode == "euler":
        logits = sample_euler(model, x_t, raw_context, cached_context, noise, ones, step_freq)
    else:
        raise NotImplementedError(f"No {mode} sampling strategy")

    if raw_output:
        return logits

    eos_id = tokenizer.eos_token_id

    predictions = logits.argmax(-1)
    replies = tokenizer.batch_decode(truncate_predictions(predictions, eos_id), skip_special_tokens=True)
    return select_reply(replies)


def sample_euler(model, x_t, raw_context, cached_context, noise, ones, step_freq):
    diffusion_steps = model.diffusion_steps
    timesteps = range(diffusion_steps, 1, -step_freq)
    num_sigmas = len(timesteps)

    for t in timesteps:
        noise.normal_(0, 1)
        x_t, cached_context = model.euler_step(
            x_t, model.sigmas[t], raw_context, cached_context, t, max(t - step_freq, 1), num_sigmas, ones, noise
        )

    noise.normal_(0, 1)
    x_0, _ = model.euler_step(x_t, model.sigmas[1], raw_context, cached_context, 1, 0, num_sigmas, ones, noise)

    logits = model.classifier(x_0)
    return logits


def sample_ddpm(model, x_t, raw_context, cached_context, noise, ones, step_freq):
    diffusion_steps = model.diffusion_steps
    timesteps = range(diffusion_steps, 1, -step_freq)

    for t in timesteps:
        x_0, cached_context = model(
            encoder_input_ids=raw_context.input_ids,
            encoder_attention_mask=raw_context.attention_mask,
            decoder_inputs_embeds=x_t,
            time_ids=t * ones,
            context=cached_context,
        )

        sigma_t = model.sigmas[max(t - step_freq, 1)]
        noise.normal_(0, 1)
        x_t = scale_input(x_0 + sigma_t * noise, sigma_t)

    x_0, _ = model(
        encoder_attention_mask=raw_context.attention_mask,
        decoder_inputs_embeds=x_t,
        time_ids=ones,
        context=cached_context,
    )

    logits = model.classifier(x_0)
    return logits


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
