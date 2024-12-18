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

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x+(self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self,features:int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

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
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1) # batch, h, seq_len, seq_len

        if dropout is not None:
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
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward: FeedForwardBloack, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.fead_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connection[1](x, self.fead_forward)
        return x
    
class Encoder(nn.Module):

    def __init__(self,features: int,  layer: nn.ModuleList):
        super().__init__()
        self.layers = layer
        self.layer_norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)

class DecoderBlock(nn.Module):

    def __init__(self,features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBloack, dropout: float):
        super().__init__()

        self.attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x:self.attention_block(x,x,x,tgt_mask))
        x = self.residual_connection[1](x, lambda x:self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # batch seq_len, d_model ---> batch, seq_len, vocab_size
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embbeding: InputEmbedding, tgt_embbeding: InputEmbedding,
                 src_pos: PositionlEncoding, tgt_pos: PositionlEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.src_embbeding = src_embbeding
        self.tgt_embbeding = tgt_embbeding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embbeding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embbeding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_voc_size: int, tgt_voc_size: int, 
                      src_seq_len: int, tgt_seq_len: int,  
                      d_model: int = 512, N: int = 6, h: int = 8,
                      dropout: float = 0.1, d_ff: int = 2048):
    # cretaeing embbeding layer
    src_embbeding = InputEmbedding(d_model, src_voc_size)
    tgt_embbeding = InputEmbedding(d_model, tgt_voc_size)
    
    # pos encding layer
    src_pos = PositionlEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionlEncoding(d_model, tgt_seq_len, dropout)

    # creat the encoder blocks
    encoder_blocks = []
    for _ in range(N): 
        encoder_self_atten_block = MultiHeadAttention(d_model, h, dropout)
        ff_block = FeedForwardBloack(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_atten_block, ff_block, dropout)
        encoder_blocks.append(encoder_block)
    
    #creat decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_atten_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_atten_block = MultiHeadAttention(d_model, h, dropout)
        ff_block = FeedForwardBloack(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_atten_block, decoder_cross_atten_block, ff_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # creat the ecoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # creat projection layer
    projection_layer = ProjectionLayer(d_model, tgt_voc_size)

    # creat the transformer
    transformer = Transformer(encoder, decoder, src_embbeding, tgt_embbeding, src_pos, tgt_pos, projection_layer)

    # initilaization the params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer