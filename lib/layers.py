import torch.nn as nn
import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, scatter, softmax
from torch_geometric.nn.norm import GraphNorm, BatchNorm


class OwnConv(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops=True, bias=False):
        super(OwnConv, self).__init__(aggr='add')
        self.add_self_loops = add_self_loops
        self.lin = nn.Linear(in_channels, out_channels)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        # if not self.bias:
        #     zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None, edge_atten=None):     
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype,
                                     device=x.device)
        if self.add_self_loops:
            if edge_atten is not None:
                _, edge_atten = add_remaining_self_loops(edge_index, edge_atten)
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight)

        col = edge_index[1]
        deg = scatter(edge_weight, col, dim=0)
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = edge_weight * deg_inv_sqrt[col]

        x = self.lin(x)
 
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None, edge_atten=edge_atten)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_weight, edge_atten):    
        if edge_weight is None:
            m = x_j
        else:
            m = edge_weight.view(-1, 1) * x_j

        if edge_atten is None:
            return m
        else:
            return m * edge_atten
        

class LightGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops=False, bias=False):
        super(LightGAT, self).__init__(aggr='add')
        self.add_self_loops = add_self_loops
        self.lin_src = nn.Linear(in_channels, out_channels)
        self.lin_dst = nn.Linear(in_channels, out_channels)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        # if not self.bias:
        #     zeros(self.bias)

    def forward(self, x, edge_index, edge_att=None):  
        # if self.add_self_loops:
        #     edge_index, _ = add_remaining_self_loops(edge_index)

        lat_lon = x[:, -2:]
        
        x_src = self.lin_src(x)   
        x_dst = self.lin_dst(x)   

        h = (x_src, x_dst)

        alpha = self.edge_updater(edge_index, h=h)
        out = self.propagate(edge_index, x=lat_lon, alpha=alpha, edge_att=edge_att)
        # ipdb.set_trace()

        if self.bias is not None:
            out = out + self.bias

        return out
    
    def edge_update(self, h_j, h_i, index):
        temp = h_j.shape[1] ** 0.5
        alpha = torch.sum((h_j / temp) * h_i, dim=-1)
        alpha = softmax(alpha, index)

        return alpha

    def message(self, x_j, alpha, edge_att):    
        m = alpha.unsqueeze(-1) * x_j
        if edge_att is not None:
            m = m * edge_att
        return m


class EdgeMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.mlp = nn.Sequential(*(self.layer(input_size, hidden_size) + [nn.Linear(hidden_size, output_size)]))
        # self.mlp = nn.Sequential(*(self.layer(in_features*2, in_features*4) + self.layer(in_features*4, in_features) + [nn.Linear(in_features, 1)]))
 
    def layer(self, input_size, hidden_size):
        return [nn.Linear(input_size, hidden_size), GraphNorm(hidden_size), nn.ReLU()]
        # return [nn.Linear(input_size, hidden_size), BatchNorm(hidden_size), nn.ReLU()]

    def forward(self, emb, edge_index, batch):
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        f12 = torch.cat([f1, f2], dim=-1)
        for module in self.mlp:
            if isinstance(module, GraphNorm):
                f12 = module(f12, batch[col])
            elif isinstance(module, BatchNorm):
                f12 = module(f12)
            else:
                f12 = module(f12)
        att_log_logits = f12

        return att_log_logits
    
    
class FeatureMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.mlp = nn.Sequential(*(self.layer(input_size, hidden_size) + [nn.Linear(hidden_size, output_size)]))
 
    def layer(self, input_size, hidden_size):
        return [nn.Linear(input_size, hidden_size), GraphNorm(hidden_size), nn.ReLU()]
        # return [nn.Linear(input_size, hidden_size), BatchNorm(hidden_size), nn.ReLU()]

    def forward(self, z, batch):
        for module in self.mlp:
            if isinstance(module, GraphNorm):
                z = module(z, batch)
            elif isinstance(module, BatchNorm):
                z = module(z)
            else:
                z = module(z)
        att_log_logits = z

        return att_log_logits        