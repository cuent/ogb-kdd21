import gzip
import json
import os
import random
from pathlib import Path
from typing import Dict

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


def get_difpool_model_without_pred(cfg_path):
    with open(cfg_path, "r"):
        cfg = yaml.safe_load(cfg_path)

    model = DiffPoolGNN(**cfg["args"])
    model.graph_pred_linear = Identity()
    return model


def get_linear_model_without_pred(cfg_path):
    with open(cfg_path, "r"):
        cfg = yaml.safe_load(cfg_path)

    model = LinearModel(**cfg["args"])
    model.layer_out = Identity()
    return model


MODELS = {
    "diffpool": get_difpool_model_without_pred,
    "linear": get_linear_model_without_pred,
}
DATASETS = {
    "diffpool": {"name": "dgl", "cls": DglPCQM4MDataset},
    "linear": {"mame": "linear", "cls": LinearPCQM4MDataset},
}


def get_models(
    models_cfg: Dict[str, str], device: torch.device
) -> torch.nn.ModuleDict:
    models = torch.nn.ModuleDict()
    for model_name, model_cfg in models_cfg.items():
        models[model_name] = MODELS[model_name](**model_cfg).to(device)

    return models


def get_y(
    raw_data_path: str = "data/dataset/pcqm4m_kddcup2021/raw/data.csv.gz",
):
    with gzip.open(raw_data_path, "rb") as f:
        raw_data = pd.read_csv(f)

    hg = raw_data["homolumogap"].values
    hg = torch.from_numpy(hg).view(-1, 1)
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

    models = get_models(cfg["models"], device=device)
    models_datasets = {model: DATASETS[model]["name"] for model in models}
    datasets = {}

    for model in cfg["models"]:
        model_ds = DATASETS[model]["name"]
        if model_ds not in datasets:
            if model_ds != "linear":
                datasets[model_ds] = load_dataset(DATASETS[model]["cls"])
            else:
                datasets[model_ds] = LinearPCQM4MDataset()

    datasets["y"] = get_y()
    aggregator = DatasetAggregator(datasets)
    split_idx = torch.load(
        "data/dataset/pcqm4m_kddcup2021/split_dict.pt"
    ).astype(torch.LongTensor)

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
        models=models, model_datasets=models_datasets, **cfg["args"]
    )

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
