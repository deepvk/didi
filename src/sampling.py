import torch

from src.diffusion.utils import get_euler_variables, scale_input


@torch.no_grad()
def sample(
    raw_context, model, mode, step_freq, noise_factor=1, tokenizer=None, max_len=-1, raw_output=False, skip_special=True
):
    input_ids = raw_context.input_ids
    emb = model.emb(input_ids)[:, :max_len]

    x_t = torch.randn_like(emb) * model.sigmas[-1] * noise_factor**2

    cached_context = None
    ones = torch.ones((emb.shape[0], 1), dtype=torch.long, device=emb.device)
    noise = torch.empty_like(emb)

    if mode == "ddpm":
        logits = sample_ddpm(model, x_t, raw_context, cached_context, noise, ones, step_freq)
    elif mode == "euler":
        logits = sample_euler(model, x_t, raw_context, cached_context, noise, ones, step_freq)
    else:
        raise NotImplementedError(f"No {mode} sampling strategy")

    if raw_output:
        return logits

    elif tokenizer is None:
        raise ValueError("`tokenizer` must be provided.")

    predictions = logits.argmax(-1)
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id
    replies = tokenizer.batch_decode(truncate_predictions(predictions, eos_id), skip_special_tokens=skip_special)
    return select_reply(replies)


def sample_ddpm(model, x_t, raw_context, cached_context, noise, ones, step_freq):
    diffusion_steps = model.diffusion_steps
    timesteps = range(diffusion_steps, 1, -step_freq)

    x_t = scale_input(x_t, model.sigmas[-1])

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


def sample_euler(model, x_t, raw_context, cached_context, noise, ones, step_freq):
    diffusion_steps = model.diffusion_steps
    timesteps = range(diffusion_steps, 1, -step_freq)
    num_sigmas = len(timesteps)

    for t in timesteps:
        noise.normal_(0, 1)
        x_t, cached_context = euler_step(
            model, x_t, model.sigmas[t], raw_context, cached_context, max(t - step_freq, 1), num_sigmas, ones, noise
        )

    noise.normal_(0, 1)
    x_0, _ = euler_step(model, x_t, model.sigmas[1], raw_context, cached_context, 0, num_sigmas, ones, noise)

    logits = model.classifier(x_0)
    return logits


def euler_step(model, x_t, sigma_t, raw_context, cached_context, next_t, num_sigmas, ones, noise):
    x_t, sigma_hat = get_euler_variables(x_t, noise, sigma_t, model.s_churn, model.s_tmin, model.s_tmax, num_sigmas)
    t_hat = ((model.sigmas - sigma_hat).abs()).argmin()

    x_0, cached_context = model(
        encoder_input_ids=raw_context.input_ids,
        encoder_attention_mask=raw_context.attention_mask,
        decoder_inputs_embeds=scale_input(x_t, sigma_hat),
        time_ids=t_hat * ones,
        context=cached_context,
    )

    d = (x_t - x_0) / sigma_hat
    dt = model.sigmas[next_t] - sigma_hat
    x_t = x_t + d * dt
    return x_t, cached_context


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
