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
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % h == 0, "d_model in not div bye h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #wq
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model) # w_0
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # batch, h, seq_len, d_k ---> batch, h, seq_len, seq_len
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask: 
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1) # batch, h, seq_len, seq_len

        if dropout:
            attention_score = dropout(attention_score)

        return(attention_score @ value), attention_score 


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) ---> (Batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) ---> (Batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) ---> (Batch, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # batch, h, seq_len ---> batch, seq_len, d_k ---> batch, seq_len, d_model
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        # batch, seq_len, d_model ---> batch, seq_len, d_model
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    
