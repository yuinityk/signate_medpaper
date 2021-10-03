import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class SRTitleClassifyTransformer(nn.Module):
    def __init__(self, model_name, config=None):
        super().__init__()
        self.transformer_title = AutoModelForSequenceClassification \
                                .from_pretrained(model_name, config=config)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids_title=None, attention_mask_title=None):
        transformer_output = self.transformer_title(
            input_ids=input_ids_title,
            attention_mask=attention_mask_title,
        )
        output = self.sigmoid(transformer_output.logits).squeeze(1)
        return output


class SRTitleEmbedTransformer(nn.Module):
    def __init__(self, model_name, config=None):
        super().__init__()
        self.transformer_title = AutoModelForSequenceClassification \
                                .from_pretrained(model_name, config=config)

    def forward(self, input_ids_title=None, attention_mask_title=None):
        transformer_embed = self.transformer_title.bert( # TODO deal with other architecture than BERT
            input_ids=input_ids_title,
            attention_mask=attention_mask_title,
        )
        transformer_embed = transformer_embed.last_hidden_state * torch.unsqueeze(attention_mask_title, 2)
        transformer_embed = transformer_embed.sum(axis=1) / attention_mask_title.sum(axis=1, keepdim=True)
        return transformer_embed # batch x hidden_dim (768)
