import torch
import torch.nn as nn
from torch.nn import MSELoss


class MaxMarginContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, metric='euclidean', device='cuda'):
        super().__init__()
        self.mse = MSELoss(reduction='none')
        self.margin = margin
        self.metric = metric
        self.device = device

    def forward(self, embed_0, embed_1, label_0, label_1)
        dist = None
        if self.metric == 'euclidean':
            dist = ((embed_0 - embed_1) ** 2).sum(axis=1) # size: batch

        label_diff = (label_0 != label_1).to(torch.uint8)
        zero_tensor = torch.zeros_like(label_0).to(self.device)
        loss = (1-label_diff) * dist \
             + label_diff * torch.maximum(zero_tensor, self.margin - dist**0.5) ** 2

        return loss


