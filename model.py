import torch 
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_module: int, vocab_size: int):
        super().__init__()
        self.d_module = d_module
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_module)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_module)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, d_k, mask, dropout: nn.Dropout=None):
        d_k = query.shape[-1]
        
        # (batch_size, h, seq_length, d_k) --> (batch_size, h, seq_length, seq_length)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1) # (batch_size, h, seq_length, seq_length)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch_size, h, seq_length, seq_length) --> (batch_size, h, seq_length, d_k)
        attention_output = torch.matmul(attention_scores, value)
        return (attention_output) , attention_scores

    def forward(self, q, k, v, mask):
        query = self.W_Q(q)  # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        key = self.W_K(k)   # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        value = self.W_V(v) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)

        # (batch_size, seq_length, d_model) --> (batch_size, h, seq_length, d_k) --> (batch_size, h, seq_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # (batch_size, h, seq_length, d_k) --> (batch_size, h, seq_length, d_k)
        x, self.attention_scores = self.attention(query, key, value, self.d_k, mask, self.dropout)

        # (batch_size, h, seq_length, d_k) --> (batch_size, seq_length, h, d_k) --> (batch_size, seq_length, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # self.h * self.d_k = d_model

        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        return self.W_O(x)   

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection1 = ResidualConnection(dropout)
        self.residual_connection2 = ResidualConnection(dropout)

    def forward(self, x, src_mask):
        x = self.residual_connection1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection2(x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection1 = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection1[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection1[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection1[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)