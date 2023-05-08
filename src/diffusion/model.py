from functools import partial

import torch
from lightning import LightningModule
from math import sqrt
from torch import nn
from transformers import AutoModel
from transformers import BertConfig

from src.diffusion.utils import configure_schedule, get_x0, get_diffusion_variables
from src.sampling import sample
from src.metrics import calculate_batch_ce
from src.utils import zero_rank_info


def get_components(name: str, mode: str, **model_kwargs):
    model = AutoModel.from_pretrained(name)

    if mode == "same":
        return model.encoder, model.decoder, model.config.d_model

    elif mode == "bert":
        decoder_config = BertConfig(
            vocab_size=model.config.vocab_size,
            is_decoder=True,
            add_cross_attention=True,
            **model_kwargs,
        )
        zero_rank_info(f"BERT config:\n{decoder_config}")

        decoder = AutoModel.from_config(decoder_config)

        return model.encoder, decoder, decoder.config.d_model


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
        emb_dim: int,
        vocabulary_size: int,
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
        self.encoder_dim = encoder.config.d_model
        self.decoder_dim = emb_dim

        self.emb = nn.Embedding(vocabulary_size, emb_dim, padding_idx=pad_idx)
        self.time_embeds = nn.Embedding(diffusion_steps + 1, emb_dim)

        self.sampling_mode = sampling_mode
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax

        self.encoder = encoder
        freeze_params(self.encoder)

        self.decoder = decoder
        self.classifier = nn.Linear(emb_dim, vocabulary_size)

        if self.encoder_dim != self.decoder_dim:
            self.adapter = nn.Sequential(nn.Linear(self.encoder_dim, emb_dim), nn.Tanh(), nn.Linear(emb_dim, emb_dim))

        sigmas, std_0 = configure_schedule(diffusion_steps, schedule)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("std_0", std_0)

        self.lr, self.warmup, self.min_lr = lr, warmup_steps, min_lr

        self.val_ce: list[float] = []
        self.val_acc: list[float] = []
        self.batch_decoder = batch_decoder

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.0)  # Fully control LR from scheduler
        scheduler_lambda = partial(rsqrt_with_warmup, max_lr=self.lr, min_lr=self.min_lr, warmup=self.warmup)
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda),
            "interval": "step",
        }
        return [optimizer], [lr_scheduler_config]

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

        if context is None:
            with torch.no_grad():
                context = self.encoder(
                    input_ids=encoder_input_ids, attention_mask=encoder_attention_mask
                ).last_hidden_state

                if self.encoder_dim != self.decoder_dim:
                    context = self.adapter(context)

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

        logits = self.classifier(x_0)  # [batch size; seq len; vocab size]
        ce = calculate_batch_ce(logits, target.input_ids, target.attention_mask)

        non_pad_mask = target.attention_mask.unsqueeze(-1)
        mse = torch.where(
            t == 1,
            flat_mean((x_0_hat - emb) ** 2 * non_pad_mask),
            flat_mean((x_0_hat - x_0) ** 2 * non_pad_mask),
        ).mean()

        t0_loss = (x_0**2 * non_pad_mask).mean()
        loss = mse + ce + t0_loss

        metrics = {"train/mse": mse, "train/ce": ce, "train/t0": t0_loss, "train/loss": loss}
        self.log_dict(metrics, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: list, batch_idx: int):
        raw_context, target = batch
        logits = sample(raw_context, self, self.sampling_mode, self.step_freq, raw_output=True)
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
            self.logger.log_text("samples", columns=["context", "reply", "predictions"], data=data)

    def on_validation_epoch_end(self):
        metrics = {"val/ce": sum(self.val_ce) / len(self.val_ce), "val/accuracy": sum(self.val_acc) / len(self.val_acc)}
        self.log_dict(metrics, sync_dist=True, on_step=False, on_epoch=True)
        self.val_ce.clear()
        self.val_acc.clear()


def rsqrt_with_warmup(step: int, max_lr: float, min_lr: float, warmup: int) -> float:
    """Scheduler for learning rate with a form of reverse sqrt (known as Noam favorite scheduler):
        `lr_t = max_lr * sqrt(1 / t)`

    Warm-up increases learning rate from 0 with square root form and then smoothly decay with reverse square root.
        `lr_t = max_lr * sqrt(t / warmup)` if t <= warmup
        `lr_t = max_lr * sqrt(warmup / t)` if t > warmup

    Also, there is control of minimum learning rate

    :param step: current step
    :param max_lr: maximum learning rate
    :param min_lr: minimum learning rate
    :param warmup: number of warmup steps
    :return: next learning rate
    """
    if warmup == 0:
        lr = max_lr * sqrt(1 / step)
    elif step < warmup:
        lr = max_lr * sqrt(step / warmup)
    else:
        lr = max_lr * sqrt(warmup / step)

    if min_lr is not None:
        lr = max(lr, min_lr)
    return lr
