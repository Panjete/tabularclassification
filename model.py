import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_layers=2):
        super(Model, self).__init__()
        self.class_token = torch.nn.Parameter(torch.randn(1, 100))        # dim 0 -> batch, dim 1 -> word num, dim 2 -> embedding dimension
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=4) ## 20 sets the maximum question length
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=None, enable_nested_tensor=True, mask_check=True)

    def forward(self, x):
        x = torch.cat((self.class_token.data, x), dim = 0)
        x = self.encoder(x)
        CLS_encoded = x[0:1, :]
        return CLS_encoded