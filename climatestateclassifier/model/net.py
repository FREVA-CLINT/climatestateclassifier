from torch import nn
from .encoder import Encoder
from .decoder import Decoder

from .. import config as cfg


class ClassificationNet(nn.Module):
    def __init__(self, img_sizes, in_channels, enc_dims, dec_dims, stride=(1, 1), bn=False,
                 activation=True, n_classes=2):
        super(ClassificationNet, self).__init__()
        enc_out_dim = (img_sizes[0] // (2 ** len(enc_dims)) * (img_sizes[1] // (2 ** len(enc_dims)))) * enc_dims[-1]
        self.encoder = Encoder(encoder_dims=enc_dims, in_channels=in_channels, stride=stride, bn=bn,
                               activation=activation)
        self.decoder = Decoder(encoder_out_dim=enc_out_dim, decoder_dims=dec_dims, n_classes=n_classes)

    def forward(self, input):
        # create lists for skip connections
        encoded_image = self.encoder(input)
        return self.decoder(encoded_image)
