import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class SRTitleClassifyTransformer(nn.Module):
    def __init__(self, model_name, config=None):
        super().__init__()
        self.transformer_title = AutoModelForSequenceClassification \
                                .from_pretrained(model_name, config=config, num_labels=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids_title=None, attention_mask_title=None):
        transformer_output = self.transformer_title(
            input_ids=input_ids_title,
            attention_mask=attention_mask_title,
        )
        output = self.sigmoid(transformer_output.logits).squeeze(1)
        return output
