from typing import Dict

import numpy as np
import torch.nn
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, num_features: int, out_features: int = 16):
        super(LinearModel, self).__init__()

        self.linear = nn.Linear(num_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.layer_out = nn.Linear(out_features, 1)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.bn(self.liner(inputs)))
        x = self.layer_out(x)
        return x


class AggregatedModel(nn.Module):
    def __init__(
        self,
        models: torch.nn.ModuleDict,
        model_datasets: Dict[str, str],
        linear_features: Dict[str, Dict[str, int]],
        output_features: int = 300,
    ):
        super().__init__()
        self.model_datasets = model_datasets
        self.models = models

        self.model_linears = torch.nn.ModuleList(
            {
                model_name: torch.nn.Linear(
                    in_features=num_features["in"],
                    out_features=num_features["out"],
                )
                for model_name, num_features in linear_features.items()
            }
        )

        sum_in_features = np.sum([it["out"] for it in linear_features.values()])
        self.output_features = torch.nn.Linear(
            in_features=int(sum_in_features), out_features=output_features
        )
        self.output_features_bn = torch.nn.BatchNorm1d(output_features)
        self.pred_layer = torch.nn.Linear(
            in_features=output_features, out_features=1
        )
        self.relu = nn.ReLU()

    def forward(self, batch):
        outs = []
        for model, layer in self.models:
            layer_outs = layer(batch[self.model_datasets[model]])
            layer_outs = self.model_linears[model](layer_outs)
            outs.append(layer_outs)

        outs = torch.cat(outs)
        outs = self.output_features_bn(self.relu(self.output_features(outs)))
        return self.pred_layer(outs)
