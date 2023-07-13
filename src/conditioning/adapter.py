import torch
from lightning import LightningModule
from torch import nn

from src.diffusion.model import DiDi
from src.diffusion.utils import get_diffusion_variables, get_x0
from src.pipeline.utils import calculate_train_step, freeze_params, get_cached_content, get_optimizers


class AdapterBlock(nn.Module):
    def __init__(self, input_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask):
        query = self.query(hidden_states)
        key = self.key(encoder_hidden_states)
        value = self.value(encoder_hidden_states)
        return self.attention(key, query, value, need_weights=False, key_padding_mask=encoder_attention_mask > 0)


class Adapter(LightningModule):
    def __init__(self, didi: DiDi, lr: float = 0.001, warmup_steps: int = 1, min_lr: float = None):
        super().__init__()
        self.didi = didi
        freeze_params(self.didi)

        self.decoder_layers = []
        adapter_layers = []
        for layer in didi.decoder.encoder.layer:
            self.decoder_layers.append(layer)
            adapter_layers.append(AdapterBlock(layer.output.dense.out_features, 1))

        self.adapter_layers = nn.ModuleList(adapter_layers)

        self.lr, self.warmup, self.min_lr = lr, warmup_steps, min_lr

    def configure_optimizers(self):
        return get_optimizers(self)

    def forward(
        self,
        encoder_input_ids: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        decoder_inputs_embeds: torch.Tensor = None,
        condition_input_ids: torch.Tensor = None,
        condition_attention_mask: torch.Tensor = None,
        time_ids: torch.Tensor = None,
        context: torch.Tensor = None,
        condition: torch.Tensor = None,
    ):
        if encoder_input_ids is None and context is None:
            raise ValueError("Either `encoder_input_ids` or `context` must be provided.")

        if condition_input_ids is None and condition is None:
            raise ValueError("Either `condition_input_ids` or `condition` must be provided.")

        context = context or get_cached_content(self.didi, encoder_input_ids, encoder_attention_mask)
        condition = condition or get_cached_content(self.didi, condition_input_ids, condition_attention_mask)

        time_embeds = self.didi.time_embeds(time_ids)
        hidden_states = decoder_inputs_embeds + time_embeds

        for decoder_layer, adapter_layer in zip(self.decoder_layers, self.adapter_layers):
            output = decoder_layer(
                hidden_states=hidden_states,
                encoder_hidden_states=context,
                encoder_attention_mask=encoder_attention_mask,
            )[0]
            hidden_states = adapter_layer(
                hidden_states=output,
                encoder_hidden_states=condition,
                encoder_attention_mask=condition_attention_mask,
            )[0]

        return hidden_states, context, condition

    def training_step(self, batch: list, batch_idx: int):
        raw_context, target, condition = batch
        emb = self.didi.emb(target.input_ids)
        x_0 = get_x0(emb, self.didi.std_0)
        noise = torch.randn_like(x_0)

        # x: [batch size; seq len; emb dim], t: [batch size]
        x_t, t = get_diffusion_variables(self.didi.diffusion_steps, x_0, self.didi.sigmas, noise)

        x_0_hat, *_ = self(
            encoder_input_ids=raw_context.input_ids,
            encoder_attention_mask=raw_context.attention_mask,
            decoder_inputs_embeds=x_t,
            time_ids=t,
            condition_input_ids=condition.input_ids,
            condition_attention_mask=condition.attention_mask,
        )  # [batch size; seq len; emb dim]

        loss, metrics = calculate_train_step(self.didi, emb, x_0, x_0_hat, target, t)
        self.log_dict(metrics, sync_dist=True, on_step=True, on_epoch=False)
        return loss
