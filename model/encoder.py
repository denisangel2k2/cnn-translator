import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        self.conv=nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm=nn.LayerNorm(channels)
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        residual=x
        
        out = self.conv(x)
        out = out.permute(0, 2, 1)             
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)

        out = self.conv(out)
        out = out.permute(0, 2, 1)              
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)

        out += residual
        return out
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, channels, num_layers, dropout=0.2):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.input_projection = nn.Linear(embedding_dim, channels)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, src):
        # src: (batch, seq_len)
        x = self.embedding(src)  # (batch, seq_len, embed_dim)
        x = self.input_projection(x)   # (batch, seq_len, channels)
        x = x.permute(0, 2, 1)   # (batch, channels, seq_len)
        x = self.residual_blocks(x)   # (batch, channels, seq_len)
        x = x.permute(0, 2, 1)   # (batch, seq_len, channels)
        return x
