from typing import Any, Dict, Tuple

import torch
import yaml
from ogb.lsc import PCQM4MEvaluator
from torch.nn import Identity

from src.dgl.models.diffpool import DiffPoolGNN
from src.pyg.models.bayesian_gnn import BayesianGNN
from src.pyg.models.gnn import GNN
from src.utils import move_to


def get_gin_virtual_model(model_args):
    model = GNN(gnn_type="gin", virtual_node=True, **model_args)
    return model


def get_gin_virtual_bnn_model(model_args):
    model = BayesianGNN(
        gnn_type="gin", virtual_node=True, last_layer_only=True, **model_args
    )
    return model


def get_diffpool_model(model_args):
    model = DiffPoolGNN(**model_args)
    return model


def load_model(
    model,
    checkpoint_path,
    test_dataloader,
    evaluator,
    eval_fn,
    device,
    name: str,
    freeze: bool = False,
    ignore_pred_layer: bool = False,
):
    if name != "dgl":
        batch = move_to(next(iter(test_dataloader)), device)
        model(batch)

    else:
        batch = next(iter(test_dataloader))[0]
        bg = batch.to(device)
        x = bg.ndata.pop("feat").to(device)
        edge_attr = bg.edata.pop("feat").to(device)
        model(bg, x, edge_attr)

    state_dict = torch.load(checkpoint_path)["model_state_dict"]
    model.load_state_dict(state_dict)

    print(
        "Loaded model score",
        eval_fn(
            model=model,
            loader=test_dataloader,
            evaluator=evaluator,
            device=device,
        ),
    )

    if ignore_pred_layer:
        model.graph_pred_linear = Identity()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


def get_models(
    cfg: Any,
    valid_loaders,
    device: torch.device,
    datasets: Dict[str, Any],
    models_cls: Dict[str, Any],
    ignore_pred_layer: bool = False,
) -> Tuple[torch.nn.ModuleDict, Dict[str, str]]:
    models = torch.nn.ModuleDict()
    models_type_mapping: Dict[str, str] = {}
    for model_cfg in cfg["models"]:
        model_name = list(model_cfg.keys())[0]
        model_type = model_cfg[model_name]["model"]
        cfg_path = model_cfg[model_name]["cfg"]
        ds_name = datasets[model_type]["name"]

        with open(cfg_path, "r") as f:
            model_args = yaml.safe_load(f)["args"]

        model = models_cls[model_type](model_args).to(device)
        load_model(
            model,
            model_cfg[model_name]["pretrained_path"],
            test_dataloader=valid_loaders[ds_name],
            evaluator=PCQM4MEvaluator(),
            eval_fn=datasets[model_type]["eval_fn"],
            device=device,
            freeze=model_cfg[model_name]["freeze"],
            name=datasets[model_type]["name"],
            ignore_pred_layer=ignore_pred_layer,
        )

        models[model_name] = model
        models_type_mapping[model_name] = model_type
    return models, models_type_mapping
