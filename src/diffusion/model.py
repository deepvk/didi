import torch
from lightning import LightningModule
from torch import nn
from transformers import AutoModel
from transformers import BertConfig

from src.diffusion.utils import configure_schedule, get_diffusion_variables, prepare_x0, get_xt
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


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * torch.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


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
        self.sigma_embeds = FourierFeatures(1, emb_dim)

        self.encoder = encoder
        freeze_params(self.encoder)

        self.decoder = decoder
        self.classifier = nn.Linear(emb_dim, vocabulary_size)

        alphas_cumprod_prev, self.sigma_0 = configure_schedule(diffusion_steps, schedule)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

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
        sigmas=None,
        context=None,
    ):
        if encoder_input_ids is None and context is None:
            raise ValueError("Either `encoder_input_ids` or `context` must be provided.")

        if context is None:
            with torch.no_grad():
                context = self.encoder(
                    input_ids=encoder_input_ids, attention_mask=encoder_attention_mask
                ).last_hidden_state

        sigmas_embeds = self.sigma_embeds(sigmas)
        input_embeds = decoder_inputs_embeds + sigmas_embeds

        output = self.decoder(
            inputs_embeds=input_embeds, encoder_hidden_states=context, encoder_attention_mask=encoder_attention_mask
        ).last_hidden_state
        return output, context

    def training_step(self, batch, batch_idx):
        context, target = batch
        emb = self.emb(target.input_ids)

        # x: [batch size; seq len; emb dim], t: [batch size]
        x_0, x_t, sigma_t, t = get_diffusion_variables(
            self.diffusion_steps, emb, self.alphas_cumprod_prev, self.sigma_0
        )
        x_0_hat, _ = self(
            encoder_input_ids=context.input_ids,
            encoder_attention_mask=context.attention_mask,
            decoder_inputs_embeds=x_t,
            sigmas=sigma_t,
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

    def validation_step(self, batch, batch_idx):
        context, target = batch
        emb = self.emb(target.input_ids)
        x_0 = prepare_x0(emb, self.sigma_0)

        cached_context = None
        ones = torch.ones(x_0.shape[0], dtype=torch.long, device=x_0.device)
        noise = torch.empty_like(x_0)
        for time in range(self.diffusion_steps, 1, -self.step_freq):
            t = ones * time
            noise.normal_(0, 1)
            x_t, sigma_t = get_xt(x_0, self.alphas_cumprod_prev, t, noise)
            x_0, cached_context = self(
                encoder_input_ids=context.input_ids,
                encoder_attention_mask=context.attention_mask,
                decoder_inputs_embeds=x_t,
                sigmas=sigma_t,
                context=cached_context,
            )

        noise.normal_(0, 1)
        x_1, sigma_1 = get_xt(x_0, self.alphas_cumprod_prev, ones, noise)
        x_0, _ = self(
            encoder_input_ids=context.input_ids,
            encoder_attention_mask=context.attention_mask,
            decoder_inputs_embeds=x_1,
            sigmas=sigma_1,
            context=cached_context,
        )

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
