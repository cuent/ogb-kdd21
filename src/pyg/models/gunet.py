from ogb.graphproppred.mol_encoder import AtomEncoder
import torch
from torch import nn
from torch_geometric import nn as tgnn
from torch_geometric.nn.models import GraphUNet
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm


class SupervisedGraphUNetModel(nn.Module):

    def __init__(self, depth: int, emb_dim: int):
        super().__init__()

        self._atom_encoder = AtomEncoder(emb_dim=emb_dim)

        self._unet = _GraphUNet(
            in_channels=emb_dim,
            hidden_channels=emb_dim // 2,
            out_channels=emb_dim,
            depth=depth,
            pool_ratios=0.5
        )

        self._pooling = tgnn.global_add_pool

        self._predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim, momentum=0.01),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.BatchNorm1d(emb_dim // 2),
            nn.Linear(emb_dim // 2, 1),
        )

    def forward(self, batched_data):
        h_node = self._unet(
            x=self._atom_encoder(batched_data.x),
            edge_index=batched_data.edge_index,
            batch=batched_data.batch,
        )

        h_graph = self._pooling(h_node, batch=batched_data.batch)

        hl_gap = self._predictor(h_graph)

        if self.training:
            return hl_gap

        return torch.clamp(hl_gap, min=0, max=50)


class _GraphUNet(GraphUNet):

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)

        # Custom sparse
        a = torch.sparse_coo_tensor(edge_index, edge_weight)
        b = torch.sparse_coo_tensor(edge_index, edge_weight).to_dense()

        c = (a @ b).to_sparse()
        
        edge_index = c.indices()
        edge_weight = c.values()

        # The `spspmm` function does not work on GPU - it yields a SIGSEGV.
        # Here we first transfer it to CPU, process it and then back to GPU.
        # Of course this slows down processing, but it is the only workaround
        # for now.
        #dev = edge_index.device
        #edge_index = edge_index.to("cpu")
        #edge_weight = edge_weight.to("cpu")

        #edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
        #                                 edge_weight, num_nodes, num_nodes,
        #                                 num_nodes)

        #edge_index = edge_index.to(dev)
        #edge_weight = edge_weight.to(dev)

        #__import__("pdb").set_trace()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
