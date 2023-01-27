from torch import nn

from transformers import AutoModel


def freeze_params(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


class Seq2SeqDiffusionTransformer(nn.Module):
    def __init__(self, name: str, vocabulary_size: int, diffusion_steps: int):
        super().__init__()

        model = AutoModel.from_pretrained(name)

        emb_dim = model.config.d_model

        self.emb = nn.Embedding(vocabulary_size, emb_dim, padding_idx=0)
        self.time_embeds = nn.Embedding(diffusion_steps + 1, emb_dim)

        self.encoder = model.encoder
        self.decoder = model.decoder

        self.classifier = nn.Linear(emb_dim, vocabulary_size)

        self.diffusion_steps = diffusion_steps

    @staticmethod
    def _freeze_params(model):
        for parameter in model.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        decoder_inputs_embeds=None,
        time_ids=None,
    ):
        context = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask).last_hidden_state

        time_embeds = self.time_embeds(time_ids)

        input_embeds = decoder_inputs_embeds + time_embeds.unsqueeze(1)

        output = self.decoder(
            inputs_embeds=input_embeds,
            encoder_hidden_states=context,
            encoder_attention_mask=encoder_attention_mask,
        ).last_hidden_state

        return output
