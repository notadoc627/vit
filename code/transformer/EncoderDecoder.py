import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_out = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_out, *args)

        return self.decoder(dec_X, dec_state)