import torch
from torch import nn


class LocationCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, output, gt):
        loss_dict = {
            'class': 0.0
        }
        loss_dict['class'] += torch.mean(self.ce(output, gt))
        return loss_dict
