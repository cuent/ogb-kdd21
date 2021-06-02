import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim
import typer
import yaml
from ogb.lsc import PCQM4MEvaluator
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import src.utils
from src.dataset import (
    load_dataset,
    get_data_loaders,
    get_dgl_dataloaders,
)
from src.pyg.models.bayesian_gnn import BayesianGNN
from src.pyg.models.gnn import GNN
from src.dgl.models.diffpool import DiffPoolGNN
from src.training.pyg import pyg_train, pyg_eval, pyg_test
from src.training.dgl_training import dgl_train, dgl_eval, dgl_test
from src.smiles import Smiles2GraphOGBConverter
from src.training.trainer import trainer

app = typer.Typer()


def get_model(
    model: str, model_args: Dict[str, Any], device: torch.device
) -> torch.nn.Module:
    if model == "gin":
        model = GNN(gnn_type="gin", virtual_node=False, **model_args)
    elif model == "gin-virtual":
        model = GNN(gnn_type="gin", virtual_node=True, **model_args)
    elif model == "gcn":
        model = GNN(gnn_type="gcn", virtual_node=False, **model_args)
    elif model == "gcn-virtual":
        model = GNN(gnn_type="gcn", virtual_node=True, **model_args)
    elif model == "gin-virtual-bnn":
        model = BayesianGNN(gnn_type="gin", virtual_node=True, **model_args)
    elif model == "diffpool":
        model = DiffPoolGNN(gnn_type="gin", virtual_node=True, **model_args)
    else:
        raise ValueError("Invalid GNN type")

    model = model.to(device)
    return model


def setup_seed() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)


def main(
    model_name: str = typer.Option(..., help="model name"),
    config_path: str = typer.Option(..., help="config path"),
    log_dir: str = typer.Option("", help="tensorboard log directory"),
    checkpoint_dir: str = typer.Option("", help="directory to save checkpoint"),
    save_test_dir: str = typer.Option(
        "", help="directory to save test submission file"
    ),
    metrics_path: str = typer.Option("", help="metrics path"),
    pyg_train_subset: bool = typer.Option(False, help="Train Subset for PyG"),
):
    device = os.getenv("CUDA_DEVICE", 0)
    num_workers = os.getenv("NUM_WORKERS", 0)
    
    # Training settings
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    setup_seed()
    device = torch.device(
        "cuda:" + str(device) if torch.cuda.is_available() else "cpu"
    )

    # Automatic dataloading and splitting
    smiles2graph = Smiles2GraphOGBConverter()

    dataset_cls = src.utils.get_module_from_str(cfg["dataset"])
    dataset = load_dataset(dataset_cls, smiles2graph)

    split_idx = dataset.get_idx_split()

    # automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    if model_name != "diffpool":
        train_loader, valid_loader, test_loader = get_data_loaders(
            dataset=dataset,
            split_idx=split_idx,
            num_workers=num_workers,
            save_test_dir=save_test_dir,
            train_subset=pyg_train_subset,
            **cfg["data_loader_args"],
        )
    else:
        train_loader, valid_loader, test_loader = get_dgl_dataloaders(
            dataset=dataset, num_workers=num_workers, **cfg["data_loader_args"]
        )

    if checkpoint_dir != "":
        os.makedirs(checkpoint_dir, exist_ok=True)

    model = get_model(model_name, model_args=cfg["args"], device=device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, **cfg["step_lr"])

    writer = None
    if log_dir != "":
        writer = SummaryWriter(log_dir=log_dir)

    epochs = cfg["learning_args"]["epochs"]
    reg = src.utils.get_module_from_str(cfg["reg"])()

    metrics = trainer(
        model=model,
        model_name=model_name,
        train_fn=pyg_train if model_name != "diffpool" else dgl_train,
        eval_fn=pyg_eval if model_name != "diffpool" else dgl_eval,
        test_fn=pyg_test if model_name != "diffpool" else dgl_test,
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
