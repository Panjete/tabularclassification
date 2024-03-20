import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, num_layers=2):
        super(Model, self).__init__()
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, 100, requires_grad=True))        # dim 0 -> batch, dim 1 -> word num, dim 2 -> embedding dimension
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=4)                    ## 20 sets the maximum question length
        self.encoder_layer.self_attn = nn.MultiheadAttention(100, 4, batch_first=True)
        self.encoder   = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=None, enable_nested_tensor=True, mask_check=True)
        self.PE_module = PositionalEncoding(100)
        self.fc1       = torch.nn.Linear(100, 10)
        self.relu1     = torch.nn.LeakyReLU()
        self.fc2       = torch.nn.Linear(10, 1)
        self.sigmoid   = torch.nn.Sigmoid()

    def forward(self, x):
        # x = (1, len(Q + C), 100)
        #print("X input shape = ", x.size())
        x = torch.cat((self.class_token.data, x), dim = 1) # (1, len Q + C + 1, 100)
        x = x * math.sqrt(100)
        x = self.PE_module(x)                     # (1, len Q + C + 1, 100)
        x = self.encoder(x)                       # (1, len Q + C + 1, 100)

        x = x[:, 0:1, :]                   # (1, 1, 100)
        x = torch.squeeze(x)               # (100,)
        x = self.fc1(x)                    # (10,)
        x = self.relu1(x)                  # (10,)
        x = self.fc2(x)                    # (1,)
        x = self.sigmoid(x)                # Comment if BCEWithLogitLoss       

        return x


class PositionalEncoding(nn.Module):
    ### Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)           ## Static Array, values don't change
        pe[:, :, 0::2] = torch.sin(position * div_term) ## Fills all even entries
        pe[:, :, 1::2] = torch.cos(position * div_term)
        print("PE DIMENSION = ", pe.size())             # PE = (1, 200, 100)
        self.register_buffer('pe', pe)        ## Don't learn, but save as model_dict()

    def forward(self, x):
        #print("CURRENYLY MY ASS WAS truncating to : ", x.size(1))
        #print("X before positional encodings = ", x.size(), x)
        #print("positional encodings = ", self.pe[:, :x.size(1), :].size(), self.pe[:, :x.size(1), :])
        x = x + self.pe[:, :x.size(1), :] ## 2nd part broadcasts into (1, (len question + column), 100)
        #print("X after positional encodings = ", x)
        return x