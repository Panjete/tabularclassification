import torch
import torch.nn as nn
from models import Model, ModelRowSel

## this class is used to incorporate both column selector and row selector as one module

class CM(nn.Module):
    def __init__(self, col_stacks, row_stacks):
        super(CM, self).__init__()
        self.model_col = Model(col_stacks)
        self.model_col.eval()
        self.model_row = ModelRowSel(row_stacks)
        self.model_row.eval()
        

    def load_combined(self, comb_filename):
        self.load_state_dict(torch.load(comb_filename))
        return


    def load_individual(self, col_filename, row_filename):
        self.model_col.load_state_dict(torch.load(col_filename))
        self.model_row.load_state_dict(torch.load(row_filename))
        return
    
    def save_comb(self, comb_filename):
        torch.save(self.state_dict(), comb_filename)
        return
    
    def save_indiv(self, col_filename, row_filename):
        torch.save(self.model_col.state_dict(), col_filename)
        torch.save(self.model_row.state_dict(), row_filename)
        return

    
     

