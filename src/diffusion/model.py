import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoModel

from src.diffusion.utils import configure_schedule
from src.diffusion.utils import flat_mean
from src.diffusion.utils import get_diffusion_variables
from src.diffusion.utils import get_xt
from src.diffusion.utils import prepare_x0
from src.metrics import calculate_batch_ce


def get_components(name):
    model = AutoModel.from_pretrained(name)

    return model.encoder, model.decoder, model.config.d_model


def freeze_params(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


class DiDi(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        emb_dim: int,
        vocabulary_size: int,
        diffusion_steps: int,
        schedule: str,
        step_freq: int = 10,
        lr: float = 0.0001,
    ):
        super().__init__()

        self.diffusion_steps = diffusion_steps

        self.emb = nn.Embedding(vocabulary_size, emb_dim, padding_idx=0)
        self.time_embeds = nn.Embedding(diffusion_steps + 1, emb_dim)

        self.encoder = encoder
        self.decoder = decoder

        self.classifier = nn.Linear(emb_dim, vocabulary_size)

        self.alphas_cumprod_prev = configure_schedule(diffusion_steps, schedule)

        self.step_freq = step_freq
        self.lr = lr

        freeze_params(self.encoder)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.encoder.eval()
        return self

    def eval(self):
        self.train(False)
        self.encoder.eval()
        return self

    def forward(
        self,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        decoder_inputs_embeds=None,
        time_ids=None,
    ):
        with torch.no_grad():
            context = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask).last_hidden_state

        time_embeds = self.time_embeds(time_ids)

        input_embeds = decoder_inputs_embeds + time_embeds.unsqueeze(1)

        output = self.decoder(
            inputs_embeds=input_embeds,
            encoder_hidden_states=context,
            encoder_attention_mask=encoder_attention_mask,
        ).last_hidden_state

        return output

    def training_step(self, batch, batch_idx):
        context, gt = batch

        emb = self.emb(gt)

        x_0, x_t, t = get_diffusion_variables(self.diffusion_steps, emb, self.alphas_cumprod_prev)

        x_0_hat = self(
            encoder_input_ids=context,
            decoder_inputs_embeds=x_t,
            time_ids=t,
        )

        pad_mask = gt == self.emb.padding_idx
        probs = self.classifier(x_0)
        ce = calculate_batch_ce(probs, gt, pad_mask)

        emb_mask = ~pad_mask.unsqueeze(-1)

        mse = torch.where(
            t == 1,
            flat_mean((x_0_hat - emb) ** 2 * emb_mask),
            flat_mean((x_0_hat - x_0) ** 2 * emb_mask),
        ).mean()

        t0_loss = (x_0**2 * emb_mask).mean()

        loss = mse + ce + t0_loss

        metrics = {"mse": mse.item(), "ce": ce.item(), "t0": t0_loss.item(), "loss": loss.item()}

        self.log("train", metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        context, gt = batch

        emb = self.emb(gt)
        x_0 = prepare_x0(emb)

        ones = torch.ones(x_0.shape[0]).long().to(x_0.device)

        for time in range(self.diffusion_steps, 1, -self.step_freq):
            t = ones * time
            x_t = get_xt(x_0, self.alphas_cumprod_prev, t)

            x_0 = self(
                encoder_input_ids=context,
                decoder_inputs_embeds=x_t,
                time_ids=t,
            )

        x_1 = get_xt(x_0, self.alphas_cumprod_prev, ones)

        x_0 = self(
            encoder_input_ids=context,
            decoder_inputs_embeds=x_1,
            time_ids=ones,
        )

        pad_mask = gt == self.emb.padding_idx
        probs = self.classifier(x_0)
        ce = calculate_batch_ce(probs, gt, pad_mask)

        emb_mask = ~pad_mask.unsqueeze(-1)
        preds = probs.argmax(-1)

        acc = (((preds == gt) * emb_mask).sum(axis=1) / emb_mask.sum(axis=1)).mean()

        return torch.tensor([ce.item(), acc.item()])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer

    def validation_epoch_end(self, validation_step_outputs):
        all_metrics = torch.stack(validation_step_outputs).mean(axis=0)

        metrics = {"ce": all_metrics[0], "accuracy": all_metrics[1]}

        self.log("val", metrics)
