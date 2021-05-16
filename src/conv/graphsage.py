import torch
from torch import nn as nn
from torch.nn import functional as F
import dgl.function as fn

from src.layers.dgl import (
    Bundler,
    MaxPoolAggregator,
    LSTMAggregator,
    MeanAggregator
)


class BatchedGraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True,
                 mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)

        nn.init.xavier_uniform_(
            self.W.weight,
            gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj):
        num_node_per_graph = adj.size(1)
        if self.use_bn and not hasattr(self, 'bn'):
            self.bn = nn.BatchNorm1d(num_node_per_graph).to(adj.device)

        if self.add_self:
            adj = adj + torch.eye(num_node_per_graph).to(adj.device)

        if self.mean:
            adj = adj / adj.sum(-1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            h_k = self.bn(h_k)
        return h_k

    def __repr__(self):
        if self.use_bn:
            return 'BN' + super(BatchedGraphSAGE, self).__repr__()
        else:
            return super(BatchedGraphSAGE, self).__repr__()


class GraphSageLayer(nn.Module):
    """
    GraphSage layer in Inductive learning paper by hamilton
    Here, graphsage layer is a reduced function in DGL framework
    """

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, bn=False, bias=True):
        super(GraphSageLayer, self).__init__()
        self.use_bn = bn
        self.bundler = Bundler(in_feats, out_feats, activation, dropout,
                               bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_feats, in_feats,
                                                activation, bias)
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_feats, in_feats)
        else:
            self.aggregator = MeanAggregator()

    # edge_attr not used
    def forward(self, g, h):
        h = self.dropout(h)
        g.ndata['h'] = h
        if self.use_bn and not hasattr(self, 'bn'):
            device = h.device
            self.bn = nn.BatchNorm1d(h.size()[1]).to(device)
        g.update_all(fn.copy_src(src='h', out='m'), self.aggregator,
                     self.bundler)
        if self.use_bn:
            h = self.bn(h)
        h = g.ndata.pop('h')
        return h
