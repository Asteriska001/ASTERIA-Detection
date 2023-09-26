import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
from torch_geometric.nn import GCNConv, global_max_pool

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, dropout1):
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
        tree, inputs = data
        for idx in range(tree.num_children):
            self.forward([tree.children[idx], inputs])

        if tree.num_children == 0:
            child_c = inputs[tree.id].data.new(1, self.mem_dim).fill_(0.)
            child_h = inputs[tree.id].data.new(1, self.mem_dim).fill_(0.)
        else:
            child_c, child_h = zip(*[x.state for x in tree.children])
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.id], child_c, child_h)
        return tree.state

class Vulnerability(torch.nn.Module):
    def __init__(self, h_size, num_node_feature, num_classes, feature_representation_size, drop_out_rate, num_conv_layers):
        super(Vulnerability, self).__init__()
        self.h_size = h_size
        self.num_node_feature = num_node_feature
        self.layer_num = num_conv_layers
        self.tree_lstm = ChildSumTreeLSTM(feature_representation_size, h_size, drop_out_rate)
        
        # GRUs for feature inputs
        self.grus = nn.ModuleList([nn.GRU(feature_representation_size, h_size, batch_first=True) for _ in range(4)])
        self.gru_combine = nn.GRU(h_size, h_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(drop_out_rate)
        
        self.connect = nn.Linear(h_size * num_node_feature * 2, h_size)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList(
            [GCNConv(h_size, h_size) if i < num_conv_layers - 1 else GCNConv(h_size, num_node_feature) for i in range(num_conv_layers)]
        )
        
        self.relu = nn.ReLU(inplace=True)

    def _process_feature(self, feature, gru):
        feature = pack_sequence(feature, enforce_sorted=False)
        feature, _ = gru(feature.float())
        feature, _ = pad_packed_sequence(feature, batch_first=True)
        return feature[:, -1:, :]

    def forward(self, my_data, edge_index):
        feature_1, *features = my_data
        
        # Process tree structure
        feature_vec1 = torch.cat([self.tree_lstm([f]) for f in feature_1], dim=0).view(-1, 1, self.h_size)
        
        # Process sequences
        sequence_features = [self._process_feature(f, gru) for f, gru in zip(features, self.grus)]
        feature_input = torch.cat([feature_vec1] + sequence_features, dim=1)
        
        feature_vec, _ = self.gru_combine(feature_input)
        feature_vec = self.dropout(feature_vec).view(-1)
        feature_vec = self.connect(feature_vec)
        
        for i, gcn in enumerate(self.gcn_layers):
            feature_vec = gcn(feature_vec, edge_index)
            if i < self.layer_num - 1:
                feature_vec = self.relu(feature_vec)
        
        pooled = global_max_pool(feature_vec, torch.zeros(feature_vec.shape[0], dtype=int, device=feature_vec.device))
        return nn.Softmax(dim=1)(pooled)
