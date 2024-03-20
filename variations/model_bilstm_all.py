import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_layers=1):
        super(Model, self).__init__()
              # dim 0 -> batch, dim 1 -> word num, dim 2 -> embedding dimension
        self.encoder   = torch.nn.LSTM(100, 64, num_layers=num_layers, batch_first= True, bidirectional=True) ## Embedding length == 100 
        self.fc1       = torch.nn.Linear(256, 32)
        self.relu1     = torch.nn.LeakyReLU()
        self.fc2       = torch.nn.Linear(32, 1)

    def forward(self, x):
        # x == (batch, l_of_sequence, embed_dim) == (N, L, D). here N = BatchSize, L = Length of Seq = 20, D = 64
        #print("X input shape = ", x.size())
        
        x, _ = self.encoder(x)                            # (N, L, 2*D)
        #print("X encoder output shape = ", x.size())
        x_first = x[:, 0, :].squeeze()                # (N, 2*D)
        x_last  = x[:, -1, :].squeeze()               # (N, 2*D)
        #print("X first last shape = ", x_first.size(), x_last.size())
        if self.training:
            #print("TRAINING")
            x = torch.cat((x_first, x_last), dim=1)        # (N, 4*D) 
        else:
            #print("TESTING")
            x = torch.cat((x_first, x_last), dim=0)        # (N, 4*D) 
        #print("X concat embed = ", x.size())

        x = self.fc1(x)                    # (N, 32)
        #print("X FC1 output = ", x.size())
        x = self.relu1(x)                  # (N, 32)
        x = self.fc2(x)                    # (N, 1)
        #print("X final output shape = ", x.size())     

        return x


