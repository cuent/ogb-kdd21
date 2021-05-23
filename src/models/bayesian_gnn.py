import torch
import torchbnn as bnn

from src.conv.bayesian_gnn import Bayesian_GNN_node_Virtualnode
from src.models.gnn import GNN


class BayesianGNN(GNN):
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
        super(BayesianGNN, self).__init__(
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

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = Bayesian_GNN_node_Virtualnode(
                num_layers,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            raise Exception("not implemented")

        # KL-divergence loss for Bayesian Neural Network
        self.kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)
        self.kl_weight = 0.01

        # change graph_pred_linear
        if graph_pooling == "set2set":
            embedding_dim = 2 * self.emb_dim
        else:
            embedding_dim = self.emb_dim

        self.graph_pred_linear = torch.nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=embedding_dim,
                out_features=100,
            ),
            torch.nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=100,
                out_features=100,
            ),
            torch.nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=100,
                out_features=100,
            ),
            torch.nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=100,
                out_features=self.num_tasks,
            ),
        )

    def get_kl_loss(self):
        return self.kl_weight * self.kl_loss(self)
