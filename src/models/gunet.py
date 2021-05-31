from ogb.graphproppred.mol_encoder import AtomEncoder
from torch import nn
from torch_geometric import nn as tgnn
from torch_geometric.nn.models import GraphUNet
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm


class GraphUNetModel(nn.Module):

    def __init__(self, depth: int, emb_dim: int):
        super().__init__()

        self._atom_encoder = AtomEncoder(emb_dim=emb_dim)

        encoder = _GraphUNet(
            in_channels=emb_dim,
            hidden_channels=32,
            out_channels=emb_dim,
            depth=depth,
            pool_ratios=0.5
        )

        self._gae = tgnn.GAE(
            encoder=encoder,
            decoder=tgnn.InnerProductDecoder(),
        )

        self._pooling = tgnn.global_add_pool

    def forward(self, batched_data):
        # Obtain node embeddings
        z_node = self._gae.encode(
            x=self._atom_encoder(batched_data.x),
            edge_index=batched_data.edge_index,
            batch=batched_data.batch,
        )

        # Obtain whole graph embedding
        z_graph = self._pooling(z_node, batch=batched_data.batch)

        if self.training:
            # Compute autoencoder loss
            recon_loss = self._gae.recon_loss(
                z=z_node,
                pos_edge_index=batched_data.edge_index,
            )

            return z_graph, recon_loss

        return z_graph


class _GraphUNet(GraphUNet):

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)

        dev = edge_index.device
        edge_index = edge_index.to("cpu")
        edge_weight = edge_weight.to("cpu")

        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)

        edge_index = edge_index.to(dev)
        edge_weight = edge_weight.to(dev)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
