from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import torch

import torch.nn as nn
import torch.nn.functional as F

class GCNPoolBlockLayer(torch.nn.Module):
    """graph conv-pool block

    graph convolutional + graph pooling + graph readout

    :attr GCL: graph conv layer
    :attr GPL: graph pooling layer
    """
    def __init__(self,
                 input_size,layer_num, hidden_size, pooling_ratio,

                 ):
        super(GCNPoolBlockLayer, self).__init__()
        #self._config = config
        input_size = input_size #self._config.hyper_parameters.vector_length
        self.layer_num = layer_num #self._config.gnn.layer_num
        self.input_GCL = GCNConv(input_size, hidden_size) #config.gnn.hidden_size)

        self.input_GPL = TopKPooling(hidden_size,#config.gnn.hidden_size,
                                     ratio = pooling_ratio)#config.gnn.pooling_ratio)

        for i in range(self.layer_num - 1):
            setattr(self, f"hidden_GCL{i}",
                    GCNConv(hidden_size, hidden_size))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(hidden_size,
                            ratio=pooling_ratio))

    def forward(self, data):
        #geomatric formats.
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.input_GCL(x, edge_index))
        x, edge_index, _, batch, _, _ = self.input_GPL(x, edge_index, None,
                                                       batch)
        # (batch size, hidden)
        out = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        for i in range(self.layer_num - 1):
            x = F.relu(getattr(self, f"hidden_GCL{i}")(x, edge_index))
            x, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                x, edge_index, None, batch)
            out += torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        return out