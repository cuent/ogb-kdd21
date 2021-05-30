import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as tgnn
from ogb.graphproppred.mol_encoder import (
    AtomEncoder,
    BondEncoder,
    get_atom_feature_dims,
    get_bond_feature_dims,
)
from torch_geometric.utils import degree


class GraphAE(nn.Module):

    def __init__(self, num_layers: int, emb_dim: int):
        super().__init__()

        self._gae = tgnn.GAE(
            encoder=Encoder(num_layers=num_layers, emb_dim=emb_dim),
            decoder=tgnn.InnerProductDecoder(),
        )

        self._atom_decoder = AtomDecoder(input_dim=emb_dim)
        self._bond_decoder = BondDecoder(input_dim=emb_dim)

        self._pooling = tgnn.global_add_pool

    def forward(self, batched_data):
        # Obtain node embeddings
        z_node = self._gae.encode(batched_data)

        # Obtain whole graph embedding
        z_graph = self._pooling(z_node, batch=batched_data.batch)

        if self.training:
            # Compute autoencoder loss
            recon_loss = self._gae.recon_loss(
                z=z_node,
                pos_edge_index=batched_data.edge_index,
            )
            atom_feature_loss = self._atom_decoder.loss(
                preds=self._atom_decoder(z_node),
                target=batched_data.x,
            )

            z_edge = torch.cat([
                z_node[batched_data.edge_index[0]],
                z_node[batched_data.edge_index[1]],
            ], dim=1)
            bond_feature_loss = self._bond_decoder.loss(
                preds=self._bond_decoder(z_edge),
                target=batched_data.edge_attr,
            )

            ae_loss = recon_loss + atom_feature_loss + bond_feature_loss

            return z_graph, ae_loss

        return z_graph


class BondDecoder(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()

        num_cls = get_bond_feature_dims()

        self._bond_type_clf = nn.Linear(2 * input_dim, num_cls[0])
        self._bond_stereo_clf = nn.Linear(2 * input_dim, num_cls[1])
        self._is_conjugate_clf = nn.Linear(2 * input_dim, num_cls[2])

    def forward(self, z_edge):
        return (
            self._bond_type_clf(z_edge),
            self._bond_stereo_clf(z_edge),
            self._is_conjugate_clf(z_edge),
        )

    def loss(self, preds, target):
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0

        for idx, pred in enumerate(preds):
            total_loss += loss_fn(input=pred, target=target[:, idx])

        return total_loss


class AtomDecoder(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        num_cls = get_atom_feature_dims()

        self._atom_clf = nn.Linear(input_dim, num_cls[0])
        self._chirality_clf = nn.Linear(input_dim, num_cls[1])
        self._degree_clf = nn.Linear(input_dim, num_cls[2])
        self._formal_charge_clf = nn.Linear(input_dim, num_cls[3])
        self._numH_clf = nn.Linear(input_dim, num_cls[4])
        self._num_radical_e_clf = nn.Linear(input_dim, num_cls[5])
        self._hybridization_clf = nn.Linear(input_dim, num_cls[6])
        self._is_aromatic_clf = nn.Linear(input_dim, num_cls[7])
        self._is_in_ring_clf = nn.Linear(input_dim, num_cls[8])

    def forward(self, z_node):
        return (
            self._atom_clf(z_node),
            self._chirality_clf(z_node),
            self._degree_clf(z_node),
            self._formal_charge_clf(z_node),
            self._numH_clf(z_node),
            self._num_radical_e_clf(z_node),
            self._hybridization_clf(z_node),
            self._is_aromatic_clf(z_node),
            self._is_in_ring_clf(z_node),
        )

    def loss(self, preds, target):
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0

        for idx, pred in enumerate(preds):
            total_loss += loss_fn(input=pred, target=target[:, idx])

        return total_loss


class Encoder(nn.Module):

    def __init__(self, num_layers: int, emb_dim: int):
        super().__init__()

        self.num_layers = num_layers

        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # Define encoder
        self._atom_encoder = AtomEncoder(emb_dim=emb_dim)
        self._convs = nn.ModuleList(
            [
                GCNConv(input_dim=emb_dim, output_dim=emb_dim)
                for _ in range(num_layers - 1)
            ]
            + [GCNConv(input_dim=emb_dim, output_dim=emb_dim)]
        )
        self._bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features=emb_dim) for _ in range(num_layers - 1)]
            + [nn.BatchNorm1d(num_features=emb_dim)]
        )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        # Apply encoder
        x = self._atom_encoder(x)
        for idx in range(self.num_layers):
            x = self._convs[idx](x, edge_index=edge_index, edge_attr=edge_attr)
            x = self._bns[idx](x)

            if idx != self.num_layers - 1:
                x = x.relu()

        return x


class GCNConv(tgnn.MessagePassing):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.root_emb = torch.nn.Embedding(1, output_dim)
        self.bond_encoder = BondEncoder(emb_dim=output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return (
            self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)
            + F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
