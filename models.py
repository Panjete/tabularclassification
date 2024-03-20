import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self, num_layers=1):
        super(Model, self).__init__()
              # dim 0 -> batch, dim 1 -> word num, dim 2 -> embedding dimension
        self.encoder   = torch.nn.GRU(300, 64, num_layers=num_layers, batch_first= True, bidirectional=True) ## Embedding length == 100
        self.fc1       = torch.nn.Linear(256, 32)
        self.relu1     = torch.nn.LeakyReLU()
        self.fc2       = torch.nn.Linear(32, 1)
        self.sigmoid   = torch.nn.Sigmoid()

    def forward(self, x):
        # x == (batch, l_of_sequence, embed_dim) == (N, L, D). here N = BatchSize, L = Length of Seq = 20, D = 64
        
        x, _ = self.encoder(x)                        # (N, L, 2*D)
        x_first = x[:, 0, :].squeeze()                # (N, 2*D)
        x_last  = x[:, -1, :].squeeze()               # (N, 2*D)
        x = torch.cat((x_first, x_last), dim=1)       # (N, 4*D) 
        x = self.fc1(x)                    # (N, 32)
       
        x = self.relu1(x)                  # (N, 32)
        x = self.fc2(x)                    # (N, 1)
        #x = self.sigmoid(x)               # Comment if BCEWithLogitLoss  
        return x

class ModelRowSel(nn.Module):
    def __init__(self, num_layers=1):
        super(ModelRowSel, self).__init__()
              # dim 0 -> batch, dim 1 -> word num, dim 2 -> embedding dimension
        self.encoder   = torch.nn.GRU(300, 64, num_layers=num_layers, batch_first= True, bidirectional=True) ## Embedding length == 100 
        self.fc1       = torch.nn.Linear(256, 32)
        self.relu1     = torch.nn.LeakyReLU()
        self.fc2       = torch.nn.Linear(32, 1)

    def forward(self, x):
        # x == (batch, l_of_sequence, embed_dim) == (N, L, D). here N = BatchSize, L = Length of Seq = 20, D = 64
        x, _ = self.encoder(x)                            # (N, L, 2*D)
        if self.training and x.dim()==3:
            x_first = x[: , 0, :]                    # (N, 2*D)
            x_last  = x[:, -1, :]                    # (N, 2*D)
            x = torch.cat((x_first, x_last), dim=1)  # (N, 4*D) 
            
        else:
            x_first = x[0, :].squeeze()               # (N, 2*D)
            x_last  = x[-1, :].squeeze()              # (N, 2*D)
            x = torch.cat((x_first, x_last), dim=0)   # (N, 4*D) 

        x = self.fc1(x)                    # (N, 32)
        x = self.relu1(x)                  # (N, 32)
        x = self.fc2(x)                    # (N, 1)
    
        return x

class ModelCompRows(nn.Module):
    ## For Model that requires comparing "summaries" of two sequences, in this context the question and row
    def __init__(self, num_layers=1):
        super(ModelCompRows, self).__init__()
        self.encoder_question   = torch.nn.GRU(300, 64, num_layers=num_layers, batch_first= True, bidirectional=True) 
        self.encoder_row        = torch.nn.GRU(300, 64, num_layers=num_layers, batch_first= True, bidirectional=True) ## Embedding length == 300 

    def forward(self, x, y):
        # x == (batch, l_of_sequence, embed_dim) == (N, L, D). here N = BatchSize, L = Length of Seq = 20, D = 64
       
        
        x, _ = self.encoder_question(x)             # (N, L, 2*D)
        y, _ = self.encoder_row(y) 

        if x.dim()==3:
            x_f  = x[:, -1, :64]                    # (N, D)
            x_r  = x[:,  0, 64:]                    # (N, D)
            x = torch.cat((x_f, x_r), dim=1)        # (N, 2*D) 

            y_f = y[:, -1, :64]                     # (N, 2*D)
            y_r = y[:,  0, 64:]                     # (N, 2*D)
            y = torch.cat((y_f, y_r), dim=1)        # (N, 4*D)y
            
        else:
            x_f  = x[-1, :].squeeze()               # (N, 2*D)
            x_r  = x[ 0, :].squeeze()               # (N, 2*D)
            x = torch.cat((x_f, x_r), dim=0)        # (N, 4*D) 

            y_f = y[-1, :].squeeze()               # (N, 2*D)
            y_l = y[ 0, :].squeeze()               # (N, 2*D)
            y = torch.cat((y_f, y_l), dim=0)        # (N, 4*D)y
        return x, y


class ModelTransformer(nn.Module):
    def __init__(self, num_layers=2):
        super(ModelTransformer, self).__init__()
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, 300, requires_grad=True))        # dim 0 -> batch, dim 1 -> word num, dim 2 -> embedding dimension
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=4, batch_first=True)                
        self.encoder   = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.PE_module = PositionalEncoding(300)
        self.fc1       = torch.nn.Linear(300, 30)
        self.relu1     = torch.nn.LeakyReLU()
        self.fc2       = torch.nn.Linear(30, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        cls = (self.class_token.data).expand(x.size(0), -1, -1).clone()
        x = torch.cat((cls, x), dim = 1) # (1, len Q + C + 1, 100)
        x = self.PE_module(x)                     # (1, len Q + C + 1, 100)
        x = self.encoder(x)                       # (1, len Q + C + 1, 100)

        x = x[:, :1, :]                    # (B, 1, 100) Extract the CLS out
        x = torch.squeeze(x)               # (B, 100,)
        x = self.fc1(x)                    # (B, 10,)
        x = self.relu1(x)                  # (B ,10,)
        x = self.fc2(x)                    # (B, 1,)
    
        return x


class PositionalEncoding(nn.Module):
    ### Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, max_len: int = 350):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)           ## Static Array, values don't change
        pe[:, :, 0::2] = torch.sin(position * div_term) ## Fills all even entries
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)                  ## Don't learn, but save as model_dict()

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] 
        return x