import torch
import torch.nn as nn

from src.conv.diffpool import DiffPoolBatchedGraphLayer
from src.models.gnn import GNN
from src.conv.graphsage import BatchedGraphSAGE
from src.utils import batch2tensor
import torch.nn.functional as F


class DiffPoolGNN(GNN):
    def __init__(
        self,
        num_tasks=1,
        num_layers=5,
        emb_dim=300,
        gnn_type="gin",
        virtual_node=True,
        residual=False,
        drop_ratio=0,
        JK="last",
        graph_pooling="sum",
    ):
        super(DiffPoolGNN, self).__init__(
            num_tasks,
            num_layers,
            emb_dim,
            gnn_type,
            virtual_node,
            residual,
            drop_ratio,
            JK,
            graph_pooling,
        )

        # 2x number of outputs
        self.graph_pred_linear = nn.Linear(2 * self.emb_dim, self.num_tasks)
        self.first_diffpool_layer = DiffPoolBatchedGraphLayer(
            input_dim=600,  # graph embedding dimension
            assign_dim=5,  # group to 10
            output_feat_dim=600,
            activation=F.relu,
            dropout=0.0,
            aggregator_type="meanpool",
            link_pred=False,
        )

        self.gc_after_pool = BatchedGraphSAGE(600, 600)

    def forward(self, g, x, edge_attr):
        # 1. GCN: 3628x9 -> 3628x600
        g.ndata["h"] = x
        h_node = self.gnn_node((g, x, edge_attr))
        h_graph_1 = self.pool(g, h_node)
        adj, h_node = self.first_diffpool_layer(g, h_node)

        node_per_pool_graph = int(adj.size()[0] / len(g.batch_num_nodes()))
        h_node, adj = batch2tensor(adj, h_node, node_per_pool_graph)

        h_node = self.gcn_forward_tensorized(
            h_node, adj, [self.gc_after_pool], True
        )
        #         print(h_node.shape)

        h_graph_2 = torch.sum(h_node, dim=1)

        h_graph = torch.cat([h_graph_1, h_graph_2], dim=1)

        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            return torch.clamp(output, min=0, max=50)

    def gcn_forward_tensorized(self, h, adj, gc_layers, cat=False):
        block_readout = []
        for gc_layer in gc_layers:
            h = gc_layer(h, adj)
            block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=2)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block
