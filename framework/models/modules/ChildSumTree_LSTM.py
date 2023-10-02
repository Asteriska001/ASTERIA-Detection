import torch
import torch.nn as nn
from torch.nn import GRU, Dropout, ReLU
from torch.autograd import Variable as Var


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, dropout1):
        # in_dim is the input dim and mem_dim is the output dim
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.drop = nn.Dropout(dropout1)

    def node_forward(self, inputs, child_c, child_h):
        inputs = torch.unsqueeze(inputs, 0)
        child_h_sum = torch.sum(child_h, dim=0)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(self.fh(child_h) + self.fx(inputs).repeat(len(child_h), 1))
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, data):
        #print('start ChildSumTreeLSTM:')
        #print(data.shape)
        
        tree = data[0]
        inputs = data[1]
        # The inputs here are the tree structure built from class Tree and the input is a list of values with the
        # node ids as the key to store the tree values
        _ = [self.forward([tree.children[idx], inputs]) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Var(inputs[tree.id].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[tree.id].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        tree.state = self.node_forward(inputs[tree.id], child_c, child_h)
        return tree.state