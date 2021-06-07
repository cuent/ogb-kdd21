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
from ogb.lsc import DglPCQM4MDataset
from ogb.lsc import PCQM4MEvaluator
from torch.nn import Identity
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import src.utils
from src.dataset import (
    load_dataset,
    get_torch_dataloaders,
    LinearPCQM4MDataset,
    DatasetAggregator,
    AggregateCollater,
)
from src.dgl.models.diffpool import DiffPoolGNN
from src.models import AggregatedModel
from src.models import LinearModel
from src.training.pyg import pyg_train, pyg_eval, pyg_test
from src.training.trainer import trainer

app = typer.Typer()


def get_diffpool_model_without_pred(model_args):
    model = DiffPoolGNN(**model_args)
    model.graph_pred_linear = Identity()
    return model


def get_linear_model_without_pred(model_args):
    model = LinearModel(**model_args)
    model.layer_out = Identity()
    return model


MODELS = {
    "diffpool": get_diffpool_model_without_pred,
    "linear": get_linear_model_without_pred,
}
DATASETS = {
    "diffpool": {"name": "dgl", "cls": DglPCQM4MDataset},
    "linear": {"name": "linear", "cls": LinearPCQM4MDataset},
}


def get_models(cfg: Any, device: torch.device) -> torch.nn.ModuleDict:
    models = torch.nn.ModuleDict()
    for model_cfg in cfg["models"]:
        model_name = list(model_cfg.keys())[0]
        cfg_path = model_cfg[model_name]["cfg"]

        with open(cfg_path, "r") as f:
            model_args = yaml.safe_load(f)["args"]

        models[model_name] = MODELS[model_name](model_args).to(device)

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

    models = get_models(cfg, device=device)

    datasets = {}
    model_datasets = {}
    for model in cfg["models"]:
        model_name = list(model.keys())[0]
        model_ds = DATASETS[model_name]["name"]

        model_datasets[model_name] = model_ds
        if model_ds not in datasets:
            datasets[model_ds] = load_dataset(DATASETS[model_name]["cls"])

    datasets["y"] = get_y()
    aggregator = DatasetAggregator(datasets)
    split_idx = torch.load("data/dataset/pcqm4m_kddcup2021/split_dict.pt")

    split_idx = {
        k: torch.from_numpy(v).to(dtype=torch.long)
        for k, v in split_idx.items()
    }

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
        device=device,
        **cfg["args"],
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
