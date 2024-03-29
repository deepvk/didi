from enum import Enum
from functools import partial

import torch
from lightning import LightningModule
from math import sqrt
from torch import nn
from transformers import AutoModel, BertConfig, T5EncoderModel

from src.diffusion.utils import configure_schedule, get_x0, get_diffusion_variables, scale_input
from src.metrics import calculate_batch_ce
from src.sampling import sample
from src.utils import zero_rank_info


class Modes(Enum):
    BERT = "bert"
    BLENDERBOT = "blenderbot"
    T5 = "t5"


def get_mode(base_name):
    for mode in Modes:
        if mode.value in base_name:
            return mode
    raise ValueError(f"Unsupported base name: {base_name}")


def get_components(name: str, pretrained: bool = True, **model_kwargs):
    mode = get_mode(name)
    if mode is Modes.BERT:
        encoder = AutoModel.from_pretrained(name)
        enc_dim = encoder.config.hidden_size
        if not pretrained:
            encoder = encoder.apply(encoder._init_weights)
    elif mode is Modes.BLENDERBOT:
        encoder = AutoModel.from_pretrained(name).encoder
        enc_dim = encoder.config.d_model
    else:
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        encoder = T5EncoderModel.from_pretrained(name)
        enc_dim = encoder.config.d_model

    decoder_config = BertConfig(
        vocab_size=encoder.config.vocab_size,
        is_decoder=True,
        add_cross_attention=True,
        **model_kwargs,
    )
    zero_rank_info(f"BERT config:\n{decoder_config}")

    decoder = AutoModel.from_config(decoder_config)

    return encoder, decoder, enc_dim, decoder_config.hidden_size


def freeze_params(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def flat_mean(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class DiDi(LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        enc_dim: int,
        dec_dim: int,
        vocabulary_size: int,
        freeze_encoder: bool,
        *,
        diffusion_steps: int,
        schedule: str,
        step_freq: int,
        pad_idx: int,
        opt_kwargs: dict,
        noise_factor: float = 1.0,
        anchor_loss: bool = False,
        tie_weights: bool = False,
        sampling_mode: str = "ddpm",
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("+inf"),
        batch_decoder=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[encoder, decoder])
        self.diffusion_steps = diffusion_steps
        self.noise_factor = noise_factor
        self.anchor_loss = anchor_loss
        self.step_freq = step_freq

        if noise_factor != 1 and not anchor_loss:
            zero_rank_info(f"WARNING: using noise factor, but without anchor loss, probably will collapse.")

        self.encoder_dim = enc_dim
        self.decoder_dim = dec_dim
        self.pad_idx = pad_idx

        self.emb = nn.Embedding(vocabulary_size, dec_dim, padding_idx=pad_idx)
        self.time_embeds = nn.Embedding(diffusion_steps + 1, dec_dim)

        self.sampling_mode = sampling_mode
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax

        self.freeze_encoder = freeze_encoder
        self.encoder = encoder
        if self.freeze_encoder:
            freeze_params(self.encoder)

        self.decoder = decoder
        self.classifier = nn.Linear(dec_dim, vocabulary_size, bias=False)
        if tie_weights:
            self.classifier.weight = self.emb.weight

        if self.encoder_dim != self.decoder_dim:
            self.adapter = nn.Sequential(nn.Linear(self.encoder_dim, dec_dim), nn.Tanh(), nn.Linear(dec_dim, dec_dim))

        sigmas, std_0 = configure_schedule(diffusion_steps, schedule)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("std_0", std_0)
        self.sigmas: torch.Tensor

        self.opt_kwargs = opt_kwargs

        self.val_ce: list[float] = []
        self.val_acc: list[float] = []
        self.batch_decoder = batch_decoder

    def configure_optimizers(self):
        # Fully control LR from scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.0, weight_decay=self.opt_kwargs["weight_decay"])
        if self.opt_kwargs["scheduler"] == "linear":
            scheduler_lambda = partial(
                linear,
                max_lr=self.opt_kwargs["lr"],
                min_lr=self.opt_kwargs["min_lr"],
                warmup=self.opt_kwargs["warmup_steps"],
                decay_range=self.opt_kwargs["decay_range"],
            )
        else:
            scheduler_lambda = partial(
                rsqrt_with_warmup,
                max_lr=self.opt_kwargs["lr"],
                min_lr=self.opt_kwargs["min_lr"],
                warmup=self.opt_kwargs["warmup_steps"],
            )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda),
            "interval": "step",
        }
        return [optimizer], [lr_scheduler_config]

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # Keep encoder always in eval mode
        return self

    def _encode_context(self, encoder_input_ids, encoder_attention_mask):
        context = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask).last_hidden_state

        if self.encoder_dim != self.decoder_dim:
            context = self.adapter(context)
        return context

    def forward(
        self,
        encoder_input_ids: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        decoder_inputs_embeds: torch.Tensor = None,
        time_ids: torch.Tensor = None,
        context: torch.Tensor = None,
    ):
        if encoder_input_ids is None and context is None:
            raise ValueError("Either `encoder_input_ids` or `context` must be provided.")

        if context is None:
            if self.freeze_encoder:
                with torch.no_grad():
                    context = self._encode_context(encoder_input_ids, encoder_attention_mask)
            else:
                context = self._encode_context(encoder_input_ids, encoder_attention_mask)

        time_embeds = self.time_embeds(time_ids)
        input_embeds = decoder_inputs_embeds + time_embeds

        output = self.decoder(
            inputs_embeds=input_embeds,
            encoder_hidden_states=context,
            encoder_attention_mask=encoder_attention_mask,
        ).last_hidden_state
        return output, context

    def training_step(self, batch: list, batch_idx: int):
        raw_context, target = batch
        emb = self.emb(target.input_ids)
        x_0 = get_x0(emb, self.std_0)
        noise = torch.randn_like(x_0)

        # x: [batch size; seq len; emb dim], t: [batch size]
        x_t, t = get_diffusion_variables(self.diffusion_steps, x_0, self.sigmas, noise, self.noise_factor)

        x_0_hat, _ = self(
            encoder_input_ids=raw_context.input_ids,
            encoder_attention_mask=raw_context.attention_mask,
            decoder_inputs_embeds=x_t,
            time_ids=t,
        )  # [batch size; seq len; emb dim]

        if self.anchor_loss:
            logits = self.classifier(x_0_hat)  # [batch size; seq len; vocab size]
        else:
            logits = self.classifier(x_0)
        ce = calculate_batch_ce(logits, target.input_ids, target.attention_mask)

        non_pad_mask = target.attention_mask.unsqueeze(-1)
        mse = torch.where(
            t == 1,
            flat_mean((x_0_hat - emb) ** 2 * non_pad_mask),
            flat_mean((x_0_hat - x_0) ** 2 * non_pad_mask),
        ).mean()

        noise, sigma_T = torch.randn_like(x_0), self.sigmas[-1]
        x_T = scale_input(x_0, sigma_T)
        t_T_loss = (x_T**2 * non_pad_mask).mean()

        loss = mse + ce
        if not self.anchor_loss:
            loss += t_T_loss

        with torch.no_grad():
            logits_hat = self.classifier(x_0_hat)
            ce_hat = calculate_batch_ce(logits_hat, target.input_ids, target.attention_mask)

        metrics = {"train/mse": mse, "train/ce": ce, "train/t_T": t_T_loss, "train/loss": loss, "train/ce_hat": ce_hat}
        self.log_dict(metrics, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: list, batch_idx: int):
        raw_context, target = batch
        max_trg_len = target.input_ids.shape[1]
        logits = sample(raw_context, self, self.sampling_mode, self.step_freq, max_len=max_trg_len, raw_output=True)
        predictions = logits.argmax(-1)

        self.val_ce.append(calculate_batch_ce(logits, target.input_ids, target.attention_mask).item())
        self.val_acc.append(
            (((predictions == target.input_ids) * target.attention_mask).sum() / target.attention_mask.sum()).item()
        )

        if self.batch_decoder and batch_idx == 0:
            # Should be list[list[str]]
            decoded_context = self.batch_decoder(raw_context.input_ids)
            decoded_reply = self.batch_decoder(target.input_ids)
            decoded_predictions = self.batch_decoder(predictions)
            data = list(zip(decoded_context, decoded_reply, decoded_predictions))
            self.logger.log_text("samples", columns=["context", "reply", "predictions"], data=data)  # type: ignore

    def on_validation_epoch_end(self):
        metrics = {"val/ce": sum(self.val_ce) / len(self.val_ce), "val/accuracy": sum(self.val_acc) / len(self.val_acc)}
        self.log_dict(metrics, sync_dist=True, on_step=False, on_epoch=True)
        self.val_ce.clear()
        self.val_acc.clear()


def rsqrt_with_warmup(step: int, max_lr: float, min_lr: float, warmup: int) -> float:
    """Scheduler for learning rate with a form of reverse sqrt (known as Noam favorite scheduler):
        `lr_t = max_lr * sqrt(1 / t)`

    Warm-up increases learning rate from 0 with a square root form and then smoothly decays with reverse square root.
        `lr_t = max_lr * sqrt(t / warmup)` if t <= warmup
        `lr_t = max_lr * sqrt(warmup / t)` if t > warmup

    Also, there is control of the minimum learning rate

    :param step: current step
    :param max_lr: maximum learning rate
    :param min_lr: minimum learning rate
    :param warmup: number of warmup steps
    :return: next learning rate
    """
    if warmup != 0 and step < warmup:
        return max_lr * sqrt(step / warmup)

    if warmup == 0:
        lr = max_lr * sqrt(1 / step)
    else:
        lr = max_lr * sqrt(warmup / step)

    if min_lr is not None:
        lr = max(lr, min_lr)
    return lr


def linear(step: int, max_lr: float, min_lr: float, warmup: int, decay_range: int) -> float:
    if min_lr is None:
        raise ValueError("min_lr must be set for the linear scheduler")
    if step < warmup:
        return max_lr
    if step > decay_range:
        return min_lr
    lr = (max_lr - min_lr) * (decay_range - step) / (decay_range - warmup) + min_lr
    return lr
