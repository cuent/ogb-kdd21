import torch
import torch.nn.functional as F
import torchbnn as bnn
from torch_geometric.nn import global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder

from src.pyg.layers.bayesian_gin import BayesianGINConv


# Virtual GNN to generate node embedding
class Bayesian_GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layers,
        emb_dim,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        gnn_type="gin",
    ):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(Bayesian_GNN_node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual  # add residual connection or not

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        # batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        # List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == "gin":
                self.convs.append(BayesianGINConv(emb_dim))
            elif gnn_type == "gcn":
                raise Exception("not implemented yet")
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
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
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU(),
                )
            )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        # virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1)
            .to(edge_index.dtype)
            .to(edge_index.device)
        )

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            # add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(
                    F.relu(h), self.drop_ratio, training=self.training
                )

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            # update the virtual nodes
            if layer < self.num_layers - 1:
                # add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = (
                    global_add_pool(h_list[layer], batch)
                    + virtualnode_embedding
                )

                # transform virtual nodes using MLP
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](
                            virtualnode_embedding_temp
                        ),
                        self.drop_ratio,
                        training=self.training,
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](
                            virtualnode_embedding_temp
                        ),
                        self.drop_ratio,
                        training=self.training,
                    )

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        return node_representation
