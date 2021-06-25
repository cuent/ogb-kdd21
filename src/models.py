from typing import Dict

import numpy as np
import torch.nn
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, in_features: int, out_features: int = 16):
        super(LinearModel, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.layer_out = nn.Linear(out_features, 1)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        try:
            x = self.bn(self.relu(self.linear(inputs)))
        except:
            print(inputs.shape)
            print(inputs)
            breakpoint()
        x = self.layer_out(x)
        return x


class AggregatedModel(nn.Module):
    def __init__(
        self,
        models: torch.nn.ModuleDict,
        model_datasets: Dict[str, str],
        model_mapping: Dict[str, str],
        linear_features: Dict[str, Dict[str, int]],
        device: str,
        output_features: int = 300,
    ):
        super().__init__()
        self.model_datasets = model_datasets
        self.model_mapping = model_mapping
        self.models = models
        self.device = device

        self.model_linears = nn.ModuleDict(
            {
                model_name: nn.Linear(
                    in_features=num_features["in"],
                    out_features=num_features["out"],
                )
                for model_name, num_features in linear_features.items()
            }
        )
        sum_in_features = np.sum([it["out"] for it in linear_features.values()])

        self.predictor = nn.Sequential(
            nn.Linear(
                in_features=int(sum_in_features), out_features=output_features
            ),
            nn.ReLU(),
            nn.BatchNorm1d(output_features),
            nn.Linear(in_features=output_features, out_features=1),
        )

    def forward(self, batch):
        outs = []
        for name, model in self.models.items():
            ds = self.model_datasets[self.model_mapping[name]]
            if ds == "dgl":
                layer_outs = model(*batch[ds])
            else:
                layer_outs = model(batch[ds])

            layer_outs = self.model_linears[name](layer_outs)
            outs.append(layer_outs)

        outs = torch.cat(outs, dim=1).to(self.device)
        return self.predictor(outs)
