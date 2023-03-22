import torch
from lightning import LightningModule
from torch import nn
from transformers import AutoModel
from transformers import BertConfig

from src.diffusion.utils import configure_schedule
from src.diffusion.utils import flat_mean
from src.diffusion.utils import get_diffusion_variables
from src.diffusion.utils import get_xt
from src.diffusion.utils import prepare_x0
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
    ):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.pad_idx = pad_idx
        self.step_freq = step_freq

        self.emb = nn.Embedding(vocabulary_size, emb_dim, padding_idx=pad_idx)
        self.time_embeds = nn.Embedding(diffusion_steps + 1, emb_dim)

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = nn.Linear(emb_dim, vocabulary_size)

        self.alphas_cumprod_prev, self.sigma_0 = configure_schedule(diffusion_steps, schedule)

        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum

        freeze_params(self.encoder)

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
        context=None,
    ):
        if encoder_input_ids is None and context is None:
            raise ValueError("Either `encoder_input_ids` or `context` must be provided.")

        if context is None:
            with torch.no_grad():
                context = self.encoder(
                    input_ids=encoder_input_ids, attention_mask=encoder_attention_mask
                ).last_hidden_state

        time_embeds = self.time_embeds(time_ids)
        input_embeds = decoder_inputs_embeds + time_embeds.unsqueeze(1)

        output = self.decoder(
            inputs_embeds=input_embeds, encoder_hidden_states=context, encoder_attention_mask=encoder_attention_mask
        ).last_hidden_state
        return output, context

    def training_step(self, batch, batch_idx):
        context, target = batch
        emb = self.emb(target)

        x_0, x_t, t = get_diffusion_variables(self.diffusion_steps, emb, self.alphas_cumprod_prev, self.sigma_0)
        x_0_hat, _ = self(encoder_input_ids=context, decoder_inputs_embeds=x_t, time_ids=t)

        pad_mask = target == self.emb.padding_idx
        probs = self.classifier(x_0)
        ce = calculate_batch_ce(probs, target, pad_mask)

        emb_mask = ~pad_mask.unsqueeze(-1)

        mse = torch.where(
            t == 1,
            flat_mean((x_0_hat - emb) ** 2 * emb_mask),
            flat_mean((x_0_hat - x_0) ** 2 * emb_mask),
        ).mean()
        t0_loss = (x_0**2 * emb_mask).mean()
        loss = mse + ce + t0_loss

        metrics = {"train/mse": mse, "train/ce": ce, "train/t0": t0_loss, "train/loss": loss}
        self.log_dict(metrics, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        raw_context, target = batch
        emb = self.emb(target)
        x_0 = prepare_x0(emb, self.sigma_0)

        context = None
        ones = torch.ones(x_0.shape[0], dtype=torch.long, device=x_0.device)
        for time in range(self.diffusion_steps, 1, -self.step_freq):
            t = ones * time
            x_t = get_xt(x_0, self.alphas_cumprod_prev, t)
            x_0, context = self(encoder_input_ids=raw_context, decoder_inputs_embeds=x_t, time_ids=t, context=context)

        x_1 = get_xt(x_0, self.alphas_cumprod_prev, ones)
        x_0, context = self(encoder_input_ids=raw_context, decoder_inputs_embeds=x_1, time_ids=ones, context=context)

        non_pad_mask = target != self.pad_idx
        logits = self.classifier(x_0)
        predictions = logits.argmax(-1)

        self.val_ce.append(calculate_batch_ce(logits, target, ~non_pad_mask).item())
        self.val_acc.append((((predictions == target) * non_pad_mask).sum() / non_pad_mask.sum()).item())

    def on_validation_epoch_end(self):
        metrics = {"val/ce": sum(self.val_ce) / len(self.val_ce), "val/accuracy": sum(self.val_acc) / len(self.val_acc)}
        self.log_dict(metrics, sync_dist=True, on_step=False, on_epoch=True)
        self.val_ce.clear()
        self.val_acc.clear()
