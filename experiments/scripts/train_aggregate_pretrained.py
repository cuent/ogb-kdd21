import gzip
import json
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import typer
import yaml
from ogb.lsc import PygPCQM4MDataset, DglPCQM4MDataset
from ogb.lsc import PCQM4MEvaluator
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import src.utils
from src.dataset import (
    get_torch_dataloaders,
    get_tg_data_loaders,
    get_dgl_data_loaders,
    LinearPCQM4MDataset,
    DatasetAggregator,
    AggregateCollater,
    load_dataset_with_validloader,
)
from src.dgl.models.diffpool import DiffPoolGNN
from src.pyg.models.bayesian_gnn import BayesianGNN
from src.pyg.models.gnn import GNN
from src.models import AggregatedModel
from src.models import LinearModel
from src.training.pyg import pyg_train, pyg_eval, pyg_test
from src.training.dgl_training import dgl_eval
from src.training.trainer import trainer

app = typer.Typer()


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


MODELS = {
    "diffpool": get_diffpool_model,
    "linear": LinearModel,
    "gin-virtual": get_gin_virtual_model,
    "gin-virtual-bnn": get_gin_virtual_bnn_model,
}
DATASETS = {
    "diffpool": {
        "name": "dgl",
        "cls": DglPCQM4MDataset,
        "eval_fn": dgl_eval,
        "loader_fn": get_dgl_data_loaders,
    },
    "gin-virtual": {
        "name": "pyg",
        "cls": PygPCQM4MDataset,
        "eval_fn": pyg_eval,
        "loader_fn": get_tg_data_loaders,
    },
    "gin-virtual-bnn": {
        "name": "pyg",
        "cls": PygPCQM4MDataset,
        "eval_fn": pyg_eval,
        "loader_fn": get_tg_data_loaders,
    },
    "linear": {"name": "linear", "cls": LinearPCQM4MDataset},
}


def get_models(
    cfg: Any, valid_loaders, device: torch.device
) -> torch.nn.ModuleDict:
    models = torch.nn.ModuleDict()
    print(valid_loaders)
    for model_cfg in cfg["models"]:
        model_name = list(model_cfg.keys())[0]
        model_type = model_cfg[model_name]["model"]
        cfg_path = model_cfg[model_name]["cfg"]
        ds_name = DATASETS[model_type]["name"]

        with open(cfg_path, "r") as f:
            model_args = yaml.safe_load(f)["args"]

        model = MODELS[model_type](model_args).to(device)
        src.utils.load_model(
            model,
            model_cfg[model_name]["pretrained_path"],
            test_dataloader=valid_loaders[ds_name],
            evaluator=PCQM4MEvaluator(),
            eval_fn=DATASETS[model_type]["eval_fn"],
            device=device,
            freeze=model_cfg[model_name]["freeze"],
            name=DATASETS[model_type]["name"],
        )

        models[model_name] = model
    return models


def get_y(
    raw_data_path: str = "data/dataset/pcqm4m_kddcup2021/raw/data.csv.gz",
):
    with gzip.open(raw_data_path, "rb") as f:
        raw_data = pd.read_csv(f)

    hg = raw_data["homolumogap"].values
    hg = torch.from_numpy(hg)
    return hg


def setup_seed() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)


def main(
    config_path: str = typer.Option(..., help="config path"),
    log_dir: str = typer.Option("", help="tensorboard log directory"),
    checkpoint_dir: str = typer.Option("", help="directory to save checkpoint"),
    save_test_dir: str = typer.Option(
        "", help="directory to save test submission file"
    ),
    metrics_path: str = typer.Option("", help="metrics path"),
):
    device = int(os.getenv("CUDA_DEVICE", "0"))
    num_workers = int(os.getenv("NUM_WORKERS", "0"))

    # Training settings
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    setup_seed()
    device = torch.device(
        "cuda:" + str(device) if torch.cuda.is_available() else "cpu"
    )

    split_idx = torch.load("data/dataset/pcqm4m_kddcup2021/split_dict.pt")

    split_idx = {
        k: torch.from_numpy(v).to(dtype=torch.long)
        for k, v in split_idx.items()
    }

    datasets = {}
    valid_dataloaders = {}
    model_datasets = {}
    models_mapping = {}
    for model in cfg["models"]:
        model_name = list(model.keys())[0]
        print(model)
        model_type = model[model_name]["model"]
        model_ds = DATASETS[model_type]["name"]

        models_mapping[model_name] = model_type
        model_datasets[model_type] = model_ds
        if model_ds not in datasets:
            (
                datasets[model_ds],
                valid_dataloaders[model_ds],
            ) = load_dataset_with_validloader(
                loader=DATASETS[model_type]["cls"],
                split_dict=split_idx,
                dataloadern_fn=DATASETS[model_type]["loader_fn"],
            )

    models = get_models(cfg, valid_dataloaders, device=device)

    datasets["y"] = get_y()
    aggregator = DatasetAggregator(datasets)

    train_loader, valid_loader, test_loader = get_torch_dataloaders(
        dataset=aggregator,
        split_idx=split_idx,
        num_workers=num_workers,
        collate_fn=AggregateCollater(keys=datasets.keys()),
        **cfg["data_loader_args"],
    )

    # automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    model = AggregatedModel(
        models=models,
        model_datasets=model_datasets,
        model_mapping=models_mapping,
        device=device,
        **cfg["args"],
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = StepLR(optimizer, **cfg["step_lr"])

    writer = None
    if log_dir != "":
        writer = SummaryWriter(log_dir=log_dir)

    epochs = cfg["learning_args"]["epochs"]
    reg = src.utils.get_module_from_str(cfg["reg"])()

    metrics = trainer(
        model=model,
        model_name="Aggregator",
        train_fn=pyg_train,
        eval_fn=pyg_eval,
        test_fn=pyg_test,
        evaluator=evaluator,
        train_loader=train_loader,
        test_loader=test_loader,
        valid_loader=valid_loader,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        reg=reg,
        device=device,
        writer=writer,
        checkpoint_dir=checkpoint_dir,
        save_test_dir=save_test_dir,
    )

    if metrics_path != "":
        path = Path(metrics_path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w+") as f:
            json.dump(obj=metrics, fp=f)


if __name__ == "__main__":
    typer.run(main)
