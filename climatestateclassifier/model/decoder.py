import torch
from torch import nn
from .. import config as cfg


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class ONIDecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, img_size, attention_dim, decoder_dim, n_layers, encoder_dim, dropout):
        """
        :param attention_dim: size of attention network
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(ONIDecoderWithAttention, self).__init__()
        encoded_image_size = img_size // (2 ** n_layers)
        self.encoder_dim = int(img_size // 2 ** (-2)) * (encoded_image_size ** 2)
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.dropout = dropout

        encoded_image_size = img_size // (2 ** n_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.dropout = nn.Dropout(p=self.dropout)
        if cfg.add_ssi:
            self.decode_step = nn.LSTMCell(encoder_dim + 1, decoder_dim, bias=True)  # decoding LSTMCell
        else:
            self.decode_step = nn.LSTMCell(encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        if cfg.loss_criterion == 'ce':
            self.fc = nn.Linear(decoder_dim, cfg.oni_resolution)  # linear layer to find scores over vocabulary
        else:
            self.fc = nn.Linear(decoder_dim, 1)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.fc.bias.data_train.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, oni_input, ssi):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        encoder_out = self.adaptive_pool(encoder_out)
        encoder_out = encoder_out.permute(0, 2, 3, 1)

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # Create tensors to hold word predicion scores and alphas
        if cfg.loss_criterion == 'ce':
            predictions = torch.zeros(batch_size, cfg.prediction_range, cfg.oni_resolution).to(cfg.device)
        else:
            predictions = torch.zeros(batch_size, cfg.prediction_range, 1).to(cfg.device)
        alphas = torch.zeros(batch_size, cfg.prediction_range, num_pixels).to(cfg.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(cfg.prediction_range):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            if cfg.add_ssi:
                h, c = self.decode_step(torch.cat([attention_weighted_encoding, ssi], dim=1), (h, c))
            else:
                h, c = self.decode_step(attention_weighted_encoding, (h, c))
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha
        predictions = predictions.transpose(1, 2)
        return predictions


class ONIDecoder(nn.Module):
    def __init__(self, img_size, attention_dim, decoder_dim, n_layers, encoder_dim, dropout):
        super(ONIDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.dropout = dropout

        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm_step = nn.LSTMCell(encoder_dim + 1 + cfg.time_steps, decoder_dim)
        self.linear1 = nn.Linear(encoder_dim + 1 + cfg.time_steps, encoder_dim + 1 + cfg.time_steps)
        if cfg.loss_criterion == "enso":
            self.linear2 = nn.Linear(encoder_dim + 1 + cfg.time_steps, 1)
        else:
            self.linear2 = nn.Linear(encoder_dim + 1 + cfg.time_steps, decoder_dim)
        self.linear_lstm1 = nn.Linear(decoder_dim*cfg.prediction_range, decoder_dim)
        self.linear_lstm2 = nn.Linear(decoder_dim, decoder_dim)
        self.activation = nn.Sigmoid()

    def forward(self, encoder_out, oni_input, ssi):
        predictions = []

        if cfg.lstm:
            # Initialize LSTM state
            h = torch.zeros(encoder_out.shape[0], self.decoder_dim).to(cfg.device)
            c = torch.zeros(encoder_out.shape[0], self.decoder_dim).to(cfg.device)
            for t in range(cfg.prediction_range):
                h, c = self.lstm_step(torch.cat([encoder_out, oni_input, ssi], dim=1), (h, c))
                predictions.append(h)
            predictions = torch.flatten(torch.stack(predictions, dim=2), start_dim=1)
            predictions = self.linear_lstm1(predictions)
            predictions = self.activation(predictions)
            predictions = self.linear_lstm2(predictions)
        else:
            predictions = self.linear1(torch.cat([encoder_out, oni_input, ssi], dim=1))
            predictions = self.activation(predictions)
            predictions = self.linear2(predictions)
        if cfg.loss_criterion == 'ce':
            predictions = torch.reshape(predictions, (encoder_out.shape[0], cfg.oni_resolution, cfg.prediction_range))
        if cfg.loss_criterion != 'enso':
            predictions = torch.reshape(predictions, (encoder_out.shape[0], cfg.prediction_range))
        else:
            predictions = self.activation(predictions)
        return self.dropout(predictions)


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
