import torch
import torch.nn as nn
from model.decoder import Decoder
from model.encoder import Encoder

class ResCnnBiLstm(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embedding_dim=256, encoder_dim=256, decoder_dim=256,
                 num_layers=3, dropout=0.2):
        super(ResCnnBiLstm, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, embedding_dim, encoder_dim, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, decoder_dim, encoder_dim, num_layers=1, dropout=dropout)

    def forward(self, src, tgt):
        encoder_output = self.encoder(src)  # (batch, src_len, encoder_dim)
        logits = self.decoder(tgt, encoder_output)  # (batch, tgt_len, vocab_size)
        return logits