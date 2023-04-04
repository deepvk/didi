import torch
from lightning import LightningModule
from torch import nn
from transformers import AutoModel
from transformers import BertConfig

from src.diffusion.utils import configure_schedule, scale_input, get_diffusion_variables, get_euler_variables
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
        self.context = None

        sigmas = configure_schedule(diffusion_steps, schedule)
        self.register_buffer("sigmas", sigmas)

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
        encoder_input_ids=None,
        encoder_attention_mask=None,
        decoder_inputs_embeds=None,
        time_ids=None,
    ):
        if encoder_input_ids is None and self.context is None:
            raise ValueError("Either `encoder_input_ids` or `context` must be provided.")

        if self.context is None:
            with torch.no_grad():
                self.context = self.encoder(
                    input_ids=encoder_input_ids, attention_mask=encoder_attention_mask
                ).last_hidden_state

        time_embeds = self.time_embeds(time_ids)
        input_embeds = decoder_inputs_embeds + time_embeds.unsqueeze(1)

        output = self.decoder(
            inputs_embeds=input_embeds,
            encoder_hidden_states=self.context,
            encoder_attention_mask=encoder_attention_mask,
        ).last_hidden_state
        return output

    def training_step(self, batch, batch_idx):
        self.context = None

        context, target = batch
        emb = x_0 = self.emb(target.input_ids)

        # x: [batch size; seq len; emb dim], t: [batch size]
        x_t, t = get_diffusion_variables(self.diffusion_steps, x_0, self.sigmas)

        x_0_hat = self(
            encoder_input_ids=context.input_ids,
            encoder_attention_mask=context.attention_mask,
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

    def euler_step(self, x_t, sigma_t, context, t, next_t, num_sigmas, ones, noise):
        x_t = scale_input(x_t, sigma_t)

        x_0 = self(
            encoder_input_ids=context.input_ids,
            encoder_attention_mask=context.attention_mask,
            decoder_inputs_embeds=x_t,
            time_ids=t * ones,
        )

        x_t, sigma_hat = get_euler_variables(x_t, noise, sigma_t, self.s_churn, self.s_tmin, self.s_tmax, num_sigmas)

        d = (x_t - x_0) / sigma_t
        dt = self.sigmas[next_t] - sigma_hat
        x_t = x_t + d * dt
        return x_t

    def validation_step(self, batch, batch_idx):
        self.context = None

        context, target = batch
        emb = self.emb(target.input_ids)
        x_t = torch.randn_like(emb) * self.sigmas[-1]

        ones = torch.ones((emb.shape[0], 1), dtype=torch.long, device=emb.device)
        noise = torch.empty_like(emb)

        num_sigmas = len(range(self.diffusion_steps, 1, -self.step_freq))

        for t in range(self.diffusion_steps, 1, -self.step_freq):
            x_t = self.euler_step(x_t, self.sigmas[t], context, t, max(t - self.step_freq, 1), num_sigmas, ones, noise)

        x_0 = self.euler_step(x_t, self.sigmas, context, 1, 0, num_sigmas, ones, noise)

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
