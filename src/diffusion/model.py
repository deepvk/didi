import torch
from torch import nn

from transformers import AutoModel


class DiffusionTransformer(nn.Module):
    def __init__(self, name: str, diffusion_steps: int):

        super().__init__()

        model = AutoModel.from_pretrained(name)

        self.emb = self._get_embeddings(model)
        self.time_embeds = self._get_time_embeddings(model, diffusion_steps)

        self.encoder = self._get_encoder(model)
        self.decoder = model.decoder

    @staticmethod
    def _freeze_params(model):
        for parameter in model.parameters():
            parameter.requires_grad = False

    def _get_embeddings(self, model):
        emb = model.shared
        self._freeze_params(emb)

        return emb

    @staticmethod
    def _get_time_embeddings(model, input_dim: int):
        emb_dim = model.config.d_model

        return nn.Embedding(input_dim, emb_dim)

    def _get_encoder(self, model):
        encoder = model.encoder
        self._freeze_params(encoder)

        return encoder

    def get_embeddings(self, input_ids):
        return self.emb(input_ids)

    def forward(
        self,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        decoder_inputs_embeds=None,
        time_ids=None,
    ):
        context = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask).last_hidden_state

        time_embeds = self.time_embeds(time_ids)
        input_embeds = decoder_inputs_embeds + time_embeds

        decoder_attention_mask = torch.ones(input_embeds.shape[:2])

        res = self.decoder(
            inputs_embeds=input_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=context,
            encoder_attention_mask=encoder_attention_mask,
        ).last_hidden_state

        return res
