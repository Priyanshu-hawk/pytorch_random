import torch
from torch import nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionlEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model) # possitional encoder
        postion = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # seq len 1
        div_term = torch.exp(torch.arange(0,d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply for sin and cosin
        pe[:, 0::2] = torch.sin(postion * div_term)
        pe[:, 1::2] = torch.cos(postion * div_term)

        ps = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x+(self.pe[:, :x.shape[1], :]).required_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBloack(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) #w1 and b2
        
    def forward(self, x):
        # (Batch, seq_len, d_model) --->  (Batch, seq_len, d_ff) ---> (Batch, seq_len, d_model)
        return(self.linear2(self.dropout(torch.relu(self.linear1(x)))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass
