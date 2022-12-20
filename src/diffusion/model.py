import torch
from torch import nn

from transformers import AutoModel


def freeze_params(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


class Seq2SeqDiffusionTransformer(nn.Module):
    def __init__(self, name: str, vocabulary_size: int, diffusion_steps: int):
        super().__init__()

        model = AutoModel.from_pretrained(name)

        self.embeddings = model.shared
        self.time_embeds = nn.Embedding(diffusion_steps, self.emb.embedding_dim)

        self.encoder = model.encoder
        self.decoder = model.decoder

        self.classifier = nn.Linear(self.emb.embedding_dim, vocabulary_size)

        self.pad_embedding = self.embeddings(torch.tensor(self.emb.padding_idx))
        self.diffusion_steps = diffusion_steps

        self.freeze_layers(["emb", "enc"])

    def freeze_layers(self, layer_names):
        if "emb" in layer_names:
            freeze_params(self.embeddings)
        if "enc" in layer_names:
            freeze_params(self.encoder)

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

        res = self.decoder(
            inputs_embeds=input_embeds,
            encoder_hidden_states=context,
            encoder_attention_mask=encoder_attention_mask,
        ).last_hidden_state

        return res
