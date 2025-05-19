import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, decoder_hidden_size, encoder_output_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(decoder_hidden_size + encoder_output_size, decoder_hidden_size)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, src_len, encoder_output_dim)

        src_len = encoder_outputs.size(1)
        hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1) 
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  
        scores = self.v(energy).squeeze(2)  
        attn_weights = torch.softmax(scores, dim=1)  
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  
        context = context.squeeze(1)  
        return context, attn_weights
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, encoder_output_dim, num_layers=1, dropout=0.2):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim, encoder_output_dim)
        self.lstm = nn.LSTM(embedding_dim + encoder_output_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, encoder_outputs):
        batch_size, target_len = target.size()
        embedded = self.dropout(self.embedding(target)) # (batch, target_len, embed_dim)
        outputs = torch.zeros(batch_size, target_len, self.fc_out.out_features, device=target.device)
        hidden = None

        for t in range(target_len):
            embedded_t = embedded[:, t, :].unsqueeze(1)  # (batch, 1, embed_dim)

            if hidden is not None:
                h_t = hidden[0][-1]
            else:
                h_t = torch.zeros(encoder_outputs.size(0), self.lstm.hidden_size, device=encoder_outputs.device)

            context, _ = self.attention(h_t, encoder_outputs)  # (batch, encoder_output_dim)
            context = context.unsqueeze(1)  # (batch, 1, encoder_output_dim)
            lstm_input = torch.cat((embedded_t, context), dim=2) # (batch, 1, embed_dim + encoder_output_dim)
            output, hidden = self.lstm(lstm_input, hidden)
            outputs[:, t, :] = self.fc_out(output).squeeze(1)

        return outputs