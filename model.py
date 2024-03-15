import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, num_layers=2):
        super(Model, self).__init__()
        self.class_token = torch.nn.Parameter(torch.randn(1, 100))        # dim 0 -> batch, dim 1 -> word num, dim 2 -> embedding dimension
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=4) ## 20 sets the maximum question length
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=None, enable_nested_tensor=True, mask_check=True)
        self.PE_module = PositionalEncoding(100)

    def forward(self, x):
        x = torch.cat((self.class_token.data, x), dim = 0)
        x = self.PE_module(x)
        x = self.encoder(x)
        CLS_encoded = x[0:1, :]
        return CLS_encoded


class PositionalEncoding(nn.Module):
    ### Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)      ## Static Array, values don't change
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)        ## Don't learn, but save as model_dict()

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x