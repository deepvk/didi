from enum import Enum

import torch
from lightning import LightningModule
from torch import nn
from transformers import AutoModel, BertConfig, BertModel, T5EncoderModel

from src.diffusion.utils import configure_schedule, get_diffusion_variables, get_x0
from src.metrics import calculate_batch_ce
from src.pipeline.sampling import sample
from src.pipeline.utils import freeze_params, get_cached_content, get_optimizers, calculate_train_step
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
        lr: float = 0.0001,
        warmup_steps: int = 0,
        min_lr: float = None,
        sampling_mode: str = "ddpm",
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("+inf"),
        batch_decoder=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[encoder, decoder])
        self.diffusion_steps = diffusion_steps
        self.pad_idx = pad_idx
        self.step_freq = step_freq
        self.encoder_dim = enc_dim
        self.decoder_dim = dec_dim

        self.emb = nn.Embedding(vocabulary_size, dec_dim, padding_idx=pad_idx)
        self.time_embeds = nn.Embedding(diffusion_steps + 1, dec_dim)

        self.sampling_mode = sampling_mode
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax

        self.encoder = encoder
        if freeze_encoder:
            freeze_params(self.encoder)

        self.decoder = decoder
        self.classifier = nn.Linear(dec_dim, vocabulary_size)

        if self.encoder_dim != self.decoder_dim:
            self.adapter = nn.Sequential(nn.Linear(self.encoder_dim, dec_dim), nn.Tanh(), nn.Linear(dec_dim, dec_dim))

        sigmas, std_0 = configure_schedule(diffusion_steps, schedule)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("std_0", std_0)
        self.sigmas: torch.Tensor

        self.lr, self.warmup, self.min_lr = lr, warmup_steps, min_lr

        self.val_ce: list[float] = []
        self.val_acc: list[float] = []
        self.batch_decoder = batch_decoder

    def configure_optimizers(self):
        return get_optimizers(self.parameters(), self.lr, self.warmup, self.min_lr)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # Keep encoder always in eval mode
        return self

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

        context = context or get_cached_content(self.didi, encoder_input_ids, encoder_attention_mask)

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
        x_t, t = get_diffusion_variables(self.diffusion_steps, x_0, self.sigmas, noise)

        x_0_hat, _ = self(
            encoder_input_ids=raw_context.input_ids,
            encoder_attention_mask=raw_context.attention_mask,
            decoder_inputs_embeds=x_t,
            time_ids=t,
        )  # [batch size; seq len; emb dim]

        loss, metrics = calculate_train_step(self.didi, emb, x_0, x_0_hat, target, t)
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
