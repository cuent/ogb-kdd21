import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from src.dgl.layers.graphsage import BatchedGraphSAGE, GraphSageLayer
from src.loss import EntropyLoss, LinkPredLoss


class DiffPoolAssignment(nn.Module):
    def __init__(self, nfeat, nnext):
        super().__init__()
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, use_bn=True)

    def forward(self, x, adj, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l = F.softmax(s_l_init, dim=-1)
        return s_l


class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, link_pred=False, entropy=True):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.log = {}
        self.link_pred_layer = LinkPredLoss()
        self.embed = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        self.assign = DiffPoolAssignment(nfeat, nnext)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        if link_pred:
            self.reg_loss.append(LinkPredLoss())
        if entropy:
            self.reg_loss.append(EntropyLoss())

    def forward(self, x, adj, log=False):
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)
        if log:
            self.log["s"] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, anext, s_l)
        if log:
            self.log["a"] = anext.cpu().numpy()
        return xnext, anext


class DiffPoolBatchedGraphLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        assign_dim,
        output_feat_dim,
        activation,
        dropout,
        aggregator_type,
        link_pred,
    ):
        super(DiffPoolBatchedGraphLayer, self).__init__()
        self.embedding_dim = input_dim
        self.assign_dim = assign_dim
        self.hidden_dim = output_feat_dim
        self.link_pred = link_pred

        self.feat_gc = GraphSageLayer(
            input_dim, output_feat_dim, activation, dropout, aggregator_type
        )

        self.pool_gc = GraphSageLayer(
            input_dim, assign_dim, activation, dropout, aggregator_type
        )

        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        self.reg_loss.append(EntropyLoss())

    def forward(self, g, h):
        #         print("DiffPoolBatchedGraphLayer forward")
        feat = self.feat_gc(
            g, h
        )  # size = (sum_N, F_out), sum_N is num of nodes in this batch
        #         print(feat.shape)
        device = feat.device
        assign_tensor = self.pool_gc(
            g, h
        )  # size = (sum_N, N_a), N_a is num of nodes in pooled graph.
        assign_tensor = F.softmax(assign_tensor, dim=1)
        assign_tensor = torch.split(assign_tensor, g.batch_num_nodes().tolist())
        assign_tensor = torch.block_diag(
            *assign_tensor
        )  # size = (sum_N, batch_size * N_a)

        h = torch.matmul(torch.t(assign_tensor), feat)
        adj = g.adjacency_matrix(transpose=False, ctx=device)
        adj_new = torch.sparse.mm(adj, assign_tensor)
        adj_new = torch.mm(torch.t(assign_tensor), adj_new)

        if self.link_pred:
            current_lp_loss = torch.norm(
                adj.to_dense() - torch.mm(assign_tensor, torch.t(assign_tensor))
            ) / np.power(g.number_of_nodes(), 2)
            self.loss_log["LinkPredLoss"] = current_lp_loss

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, adj_new, assign_tensor)

        return adj_new, h
