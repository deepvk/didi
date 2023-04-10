import torch
from lightning import LightningModule
from torch import nn
from transformers import AutoModel
from transformers import BertConfig

from src.diffusion.utils import configure_schedule, get_x0, scale_input, get_diffusion_variables, get_euler_variables
from src.metrics import calculate_batch_ce


def get_components(
    name: str,
    mode: str,
    num_hidden_layers: int = None,
    intermediate_size: int = None,
):
    model = AutoModel.from_pretrained(name)
    emb_dim = model.config.d_model

    if mode == "same":
        return model.encoder, model.decoder, emb_dim

    elif mode == "bert":
        decoder_config = BertConfig(
            vocab_size=model.config.vocab_size,
            is_decoder=True,
            hidden_size=emb_dim,
            num_attention_heads=model.config.num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            add_cross_attention=True,
        )

        decoder = AutoModel.from_config(decoder_config)

        return model.encoder, decoder, emb_dim


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
        optimizer: str = "AdamW",
        lr: float = 0.0001,
        momentum: float = 0.95,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("+inf"),
    ):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.pad_idx = pad_idx
        self.step_freq = step_freq

        self.emb = nn.Embedding(vocabulary_size, emb_dim, padding_idx=pad_idx)
        self.time_embeds = nn.Embedding(diffusion_steps + 1, emb_dim)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax

        self.encoder = encoder
        freeze_params(self.encoder)

        self.decoder = decoder
        self.classifier = nn.Linear(emb_dim, vocabulary_size)

        sigmas, std_0 = configure_schedule(diffusion_steps, schedule)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("std_0", std_0)

        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum

        self.val_ce: list[float] = []
        self.val_acc: list[float] = []

    def configure_optimizers(self):
        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            raise NotImplementedError("Unsupported optimizer")
        return optimizer

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

    def euler_step(self, x_t, sigma_t, raw_context, cached_context, t, next_t, num_sigmas, ones, noise):
        x_t = scale_input(x_t, sigma_t)

        x_0, cached_context = self(
            encoder_input_ids=raw_context.input_ids,
            encoder_attention_mask=raw_context.attention_mask,
            decoder_inputs_embeds=x_t,
            time_ids=t * ones,
            context=cached_context,
        )

        x_t, sigma_hat = get_euler_variables(x_t, noise, sigma_t, self.s_churn, self.s_tmin, self.s_tmax, num_sigmas)

        d = (x_t - x_0) / sigma_t
        dt = self.sigmas[next_t] - sigma_hat
        x_t = x_t + d * dt
        return x_t, cached_context

    def validation_step(self, batch: list, batch_idx: int):
        raw_context, target = batch
        emb = self.emb(target.input_ids)
        x_t = torch.randn_like(emb) * self.sigmas[-1]

        cached_context = None
        ones = torch.ones((emb.shape[0], 1), dtype=torch.long, device=emb.device)
        noise = torch.empty_like(emb)

        num_sigmas = len(range(self.diffusion_steps, 1, -self.step_freq))

        for t in range(self.diffusion_steps, 1, -self.step_freq):
            noise.normal_(0, 1)
            x_t, cached_context = self.euler_step(
                x_t, self.sigmas[t], raw_context, cached_context, t, max(t - self.step_freq, 1), num_sigmas, ones, noise
            )

        noise.normal_(0, 1)
        x_0, _ = self.euler_step(x_t, self.sigmas[1], raw_context, cached_context, 1, 0, num_sigmas, ones, noise)

        logits = self.classifier(x_0)
        predictions = logits.argmax(-1)

        self.val_ce.append(calculate_batch_ce(logits, target.input_ids, target.attention_mask).item())
        self.val_acc.append(
            (((predictions == target.input_ids) * target.attention_mask).sum() / target.attention_mask.sum()).item()
        )

    def on_validation_epoch_end(self):
        metrics = {"val/ce": sum(self.val_ce) / len(self.val_ce), "val/accuracy": sum(self.val_acc) / len(self.val_acc)}
        self.log_dict(metrics, sync_dist=True, on_step=False, on_epoch=True)
        self.val_ce.clear()
        self.val_acc.clear()
