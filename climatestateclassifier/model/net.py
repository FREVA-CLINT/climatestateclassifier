from torch import nn
from .. import config as cfg


class ONINet(nn.Module):
    def __init__(self, encoder, decoder, img_size=512, in_channels=1, encoding_layers=6):
        super(ONINet, self).__init__()
        encoder_out_dim = int(img_size // 2 ** (-2))
        decoder_dim = cfg.decoder_dim

        self.encoder = encoder(img_size=img_size, in_channels=in_channels, n_layers=encoding_layers)
        self.decoder = decoder(img_size=img_size, attention_dim=cfg.attention_dim, decoder_dim=decoder_dim,
                               encoder_dim=encoder_out_dim, n_layers=encoding_layers, dropout=cfg.dropout)

    def forward(self, input, oni_input, ssi):
        # create lists for skip connections
        encoded_image = self.encoder(input)
        return self.decoder(encoded_image, oni_input, ssi)


class LocationNet(nn.Module):
    def __init__(self, encoder, decoder, img_size=512, in_channels=1, encoding_layers=6, stride=(2, 2), bn=True,
                 activation=True):
        super(LocationNet, self).__init__()
        encoder_out_dim = int(img_size // 2 ** (-2)) * ((img_size // (2 ** encoding_layers)) ** 2)

        self.encoder = encoder(img_size=img_size, in_channels=in_channels, n_layers=encoding_layers,
                               stride=stride, bn=bn, activation=activation)
        self.decoder = decoder(encoder_out_dim=encoder_out_dim, decoder_dims=cfg.decoder_dims,
                               n_classes=len(cfg.locations))

    def forward(self, input):
        # create lists for skip connections
        encoded_image = self.encoder(input)
        return self.decoder(encoded_image)
