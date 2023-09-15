import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.v = nn.Parameter(torch.randn(visible_units))
        self.h = nn.Parameter(torch.randn(hidden_units))
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units))
    
    def sample_from_p(self, p):
        return torch.relu(torch.sign(p - torch.rand(p.size())))
    
    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h
    
    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v
    
    def forward(self, v):
        p_h, _ = self.v_to_h(v)
        return p_h

class DBN(nn.Module):
    def __init__(self, sizes):
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList([RBM(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        
    def forward(self, x):
        for rbm in self.rbms:
            x = rbm(x)
        return x

# Example usage:
# sizes = [input_dim, hidden_dim1, hidden_dim2, ...]
# dbn = DBN(sizes)
