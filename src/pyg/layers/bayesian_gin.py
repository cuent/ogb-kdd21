import torch
import torch.nn.functional as F
import torchbnn as bnn
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.nn import MessagePassing


class BayesianGINConv(MessagePassing):
    def __init__(self, emb_dim):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(BayesianGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=emb_dim,
                out_features=emb_dim,
            ),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=emb_dim,
                out_features=emb_dim,
            ),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
