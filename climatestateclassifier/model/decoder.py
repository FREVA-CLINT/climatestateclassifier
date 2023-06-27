from torch import nn


class LocationDecoder(nn.Module):
    def __init__(self, encoder_out_dim, decoder_dims, n_classes=3):
        super(LocationDecoder, self).__init__()

        layers = [nn.Flatten(), nn.Linear(encoder_out_dim, decoder_dims[0])]
        dims = decoder_dims + [n_classes]
        for i in range(len(decoder_dims)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(dims[i], dims[i+1]))

        self.main = nn.ModuleList(layers)

    def forward(self, encoder_out):
        for layer in self.main:
            encoder_out = layer(encoder_out)
        return encoder_out
