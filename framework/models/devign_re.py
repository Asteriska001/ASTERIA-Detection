import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class Conv(nn.Module):

    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super(Conv, self).__init__()
        self.conv1d_1_args = conv1d_1
        self.conv1d_1 = nn.Conv1d(**conv1d_1)
        self.conv1d_2 = nn.Conv1d(**conv1d_2)

        fc1_size = get_conv_mp_out_size(fc_1_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        fc2_size = get_conv_mp_out_size(fc_2_size, conv1d_2, [maxpool1d_1, maxpool1d_2])

        # Dense layers
        self.fc1 = nn.Linear(fc1_size, 1)
        self.fc2 = nn.Linear(fc2_size, 1)

        # Dropout
        self.drop = nn.Dropout(p=0.2)

        self.mp_1 = nn.MaxPool1d(**maxpool1d_1)
        self.mp_2 = nn.MaxPool1d(**maxpool1d_2)

    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

        Z = self.mp_1(F.relu(self.conv1d_1(concat)))
        Z = self.mp_2(self.conv1d_2(Z))

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

        Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
        Y = self.mp_2(self.conv1d_2(Y))

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        res = self.drop(res)
        # res = res.mean(1)
        # print(res, mean)
        sig = torch.sigmoid(torch.flatten(res))
        return sig


class Net(nn.Module):

    def __init__(self, gated_graph_conv_args, conv_args, emb_size):
        super(Net, self).__init__()
        self.ggc = GatedGraphConv(**gated_graph_conv_args)#.to(device)
        self.conv = Conv(**conv_args,
                         fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
                         fc_2_size=gated_graph_conv_args["out_channels"])#.to(device)
        # self.conv.apply(init_weights)

    def forward(self, data_list):
        outputs = []
        for data in data_list:
            data = data.to(device)
            x, edge_index = dict(data.my_data)['x'], data.edge_index
            x = x.to(device)
            edge_index = edge_index.to(device)
            #print('x,edge_index device: ', x.device, edge_index.device)
            x = self.ggc(x, edge_index)
            x = self.conv(x, dict(data.my_data)['x'])
            outputs.append(x)
        return torch.cat(outputs, dim=0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class Devign(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):#model_config, loss_lambda):
        super(Devign, self).__init__()
        # 初始化网络结构，这里使用一个假设的Net类，它需要根据具体情况定义
        from types import SimpleNamespace
        #args = SimpleNamespace(**args)
        self.model = Net(**args)
        self.loss_lambda = 1.3e-6#loss_lambda

    def forward(self, x):
        # 前向传播逻辑
        output = self.model(x)
        return output

    def loss(self, output, target):
        # 定义损失函数，这里用二元交叉熵和L1损失的组合
        return F.binary_cross_entropy(output, target) + F.l1_loss(output, target) * self.loss_lambda
