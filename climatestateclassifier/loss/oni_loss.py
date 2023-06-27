import torch
from torch import nn
from .. import config as cfg


class ONIMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, gt):
        loss_dict = {
            'graph': 0.0
        }

        # define different loss functions from output and gt
        loss_dict['graph'] += self.mse(output.squeeze(), gt.squeeze())

        return loss_dict


class ONIL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output, gt):
        loss_dict = {
            'graph': 0.0
        }

        # define different loss functions from output and gt
        loss_dict['graph'] += self.l1(output.squeeze(), gt.squeeze())

        return loss_dict


class ONICELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, output, gt):
        loss_dict = {
            'graph': 0.0
        }
        gt = torch.squeeze(gt)
        class_index = torch.zeros_like(gt).long()
        for i in range(cfg.oni_resolution, 0, -1):
            class_index[gt <= cfg.oni_range[0] + i * ((cfg.oni_range[1] - cfg.oni_range[0]) / cfg.oni_resolution)] = i
        # define different loss functions from output and gt
        loss_dict['graph'] += torch.mean(self.ce(output, class_index))

        return loss_dict


class ONIENSOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, output, gt):
        output = output.squeeze()
        august, july, june = cfg.reverse_jja_indices
        class_index = torch.zeros_like(output).to(cfg.device)
        for i in range(gt.shape[0]):
            oni_mean = torch.stack([gt[i, -august], gt[i, -july], gt[i, -june]])
            if torch.mean(oni_mean) > 0.5:
                class_index[i] = 1.0

        loss_dict = {
            'graph': 0.0
        }
        # define different loss functions from output and gt
        loss_dict['graph'] += torch.mean(self.bce(output, class_index))

        return loss_dict