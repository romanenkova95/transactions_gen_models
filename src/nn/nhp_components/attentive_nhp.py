import math

import torch
import torch.nn as nn

MINUS_INF = -1e3

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # small change here -- we use "1" for masked element
        #scores = scores.masked_fill(mask > 0, float("-inf"))
        
        #print("max scores:", scores.max().item())
        #print("min scores:", scores.min().item())
        
        scores = scores.masked_fill(mask > 0, MINUS_INF)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_model, dropout=0.1, output_linear=False):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_head == 0
        
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
       
        self.d_input = d_input 
        self.d_model = d_model
        
        self.output_linear = output_linear
        
        if output_linear:
            self.linears = nn.ModuleList(
                [nn.Linear(self.d_input, self.d_model) for _ in range(3)] + [nn.Linear(self.d_model, self.d_model), ])
        else:
            self.linears = nn.ModuleList([nn.Linear(self.d_input, self.d_model) for _ in range(3)])

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask, output_weight=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        
        x, attn_weight = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)

        if self.output_linear:
            if output_weight:
                return self.linears[-1](x), attn_weight
            else:
                return self.linears[-1](x)
        else:
            if output_weight:
                return x, attn_weight
            else:
                return x

class SublayerConnection(nn.Module):
    # used for residual connection
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward=None, use_residual=False, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_residual = use_residual
        if use_residual:
            self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask):
        if self.use_residual:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            if self.feed_forward is not None:
                return self.sublayer[1](x, self.feed_forward)
            else:
                return x
        else:
            return self.self_attn(x, x, x, mask)