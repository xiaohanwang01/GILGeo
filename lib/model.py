import torch
import torch.nn as nn
from torch_geometric.nn.conv import *

from .layers import *

    
class GILGeo(nn.Module):
    def __init__(self, dim_in):
        super(GILGeo, self).__init__()
        self.dim_in = dim_in
        self.dim_z = dim_in
        
        self.feature_fc = nn.Linear(dim_in, dim_in-2)
        self.edge_fc = nn.Linear(dim_in*2, 1)
        self.inv_pred = nn.Linear(2, 2)
        self.mix_pred = nn.Linear(2, 2)
        self.conv = LightGAT(self.dim_in, self.dim_z)


    def sampling(self, log_logits, temp):
        random_noise = torch.empty_like(log_logits).uniform_(1e-10, 1-1e-10)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        bern = ((log_logits + random_noise) / temp).sigmoid()
        return bern
    
    def mask(self, log_logits, temp, training):
        if training:
            att = self.sampling(log_logits, temp)
        else:
            att = torch.sigmoid(log_logits)
        return att
    
    def forward(self, x, edge_index, tg_mask, batch, ptr, training=True):
        feature_log_logits = self.feature_fc(x)
        feature_mask = self.mask(feature_log_logits, temp=1, training=training)
        edge_log_logits = self.edge_fc(torch.cat([x[edge_index[0]],x[edge_index[1]]], dim=-1))
        edge_att = self.mask(edge_log_logits, temp=1, training=training)
        if training:
            xc = x[:, :-2] * feature_mask
            xs = x[:, :-2] * (1-feature_mask)
            perm_xs = xs[torch.randperm(xs.shape[0])]
            perm_x = xc+perm_xs
            xc = torch.cat([xc, x[:, -2:]], dim=-1)
            perm_x = torch.cat([perm_x, x[:, -2:]], dim=-1)
            zc = self.conv(xc, edge_index, edge_att)
            perm_z = self.conv(perm_x, edge_index, edge_att)
            xc_pred = self.inv_pred(zc[tg_mask==1])
            perm_pred = self.mix_pred(perm_z[tg_mask==1])
        else:
            xc = x[:, :-2] * feature_mask
            xc = torch.cat([xc, x[:, -2:]], dim=-1)
            zc = self.conv(xc, edge_index, edge_att)
            xc_pred = self.inv_pred(zc[tg_mask==1])
            perm_pred = None
        outputs = {
            'perm_pred': perm_pred,
            'xc_pred': xc_pred,
            'feature_mask': feature_mask,
            'edge_att': edge_att
        }

        return outputs
    

class HGLGeo(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        self.dim_z = dim_in

        self.conv = HypergraphConv(self.dim_z, self.dim_z)
        self.pred = nn.Linear(2, 2)

    def forward(self, x, hyperedge_index, tg_mask):
        z = self.conv(x, hyperedge_index)
        pred = self.pred(z[tg_mask==1])
        return pred


class LightGATGeo(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.conv = LightGAT(dim_in, dim_in)
        self.pred = nn.Linear(2, 2)
    
    def forward(self, x, edge_index, tg_mask):
        z = self.conv(x, edge_index)
        pred = self.pred(z[tg_mask==1])
        return pred