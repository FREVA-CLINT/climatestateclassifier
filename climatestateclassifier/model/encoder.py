import torch.nn as nn
import torch.nn.functional as F
from .. import config as cfg


def bound_pad(input, padding):
    input = F.pad(input, (0, 0, padding[2], 0), "constant", 0.)
    input = F.pad(input, (0, 0, 0, padding[3]), "constant", 0.)
    input = F.pad(input, (padding[0], padding[1], 0, 0), mode="circular")

    return input


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dilation=(1, 1),
                 bn=True, activation=True):
        super(EncoderBlock, self).__init__()
        self.padding = 2 * padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation)

        if cfg.global_padding:
            self.trans_pad = bound_pad
        else:
            self.trans_pad = F.pad

        if stride[0] < 2:
            self.pooling = nn.MaxPool2d(2, 2)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activation:
            self.activation = nn.ReLU()

    def forward(self, input):
        pad_input = self.trans_pad(input, self.padding)
        output = self.conv(pad_input)
        if hasattr(self, "pooling"):
            output = self.pooling(output)
        if hasattr(self, "bn"):
            output = self.bn(output)
        if hasattr(self, "activation"):
            output = self.activation(output)
        return output


class Encoder(nn.Module):
    def __init__(self, img_size, in_channels, n_layers, stride=(2, 2), bn=True, activation=True):
        super(Encoder, self).__init__()

        # initialize channels
        channels = [int(img_size // 2 ** (n_layers - i - 3)) for i in range(n_layers)]
        channels.insert(0, in_channels)

        layers = []
        for i in range(n_layers):
            layers.append(EncoderBlock(channels[i], channels[i + 1], stride=stride, bn=bn, activation=activation))
        self.main = nn.ModuleList(layers)

    def forward(self, input):
        for layer in self.main:
            input = layer(input)
        return input

