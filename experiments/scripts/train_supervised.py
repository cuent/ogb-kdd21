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
from tqdm import trange

import src.utils
from src.dataset import (
    collate_dgl,
    get_data_loaders,
    get_torch_dataloaders,
    load_dataset,
)
from src.dgl.models.diffpool import DiffPoolGNN
from src.pyg.models.bayesian_gnn import BayesianGNN
from src.pyg.models.gnn import GNN
from src.smiles import Smiles2GraphOGBConverter
from src.training.dgl_training import dgl_eval, dgl_test, dgl_train
from src.training.pyg import pyg_eval, pyg_test, pyg_train
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
    elif model == "gin-virtual-diffpool":
        model = DiffPoolGNN(gnn_type="gin", virtual_node=True, **model_args)
    else:
        raise ValueError("Invalid GNN type")

    model = model.to(device)
    return model


def uses_dgl_dataset(gnn_name: str) -> bool:
    return "diffpool" in gnn_name


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
    device = int(os.getenv("CUDA_DEVICE", "0"))
    num_workers = int(os.getenv("NUM_WORKERS", "0"))

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

    use_dgl = uses_dgl_dataset(gnn_name=model_name)

    if not use_dgl:
        train_loader, valid_loader, test_loader = get_data_loaders(
            dataset=dataset,
            split_idx=split_idx,
            num_workers=num_workers,
            train_subset=pyg_train_subset,
            **cfg["data_loader_args"],
        )
    else:
        split_idx["train"] = split_idx["train"].type(torch.LongTensor)
        split_idx["test"] = split_idx["test"].type(torch.LongTensor)
        split_idx["valid"] = split_idx["valid"].type(torch.LongTensor)
        train_loader, valid_loader, test_loader = get_torch_dataloaders(
            dataset=dataset,
            num_workers=num_workers,
            split_idx=split_idx,
            collate_fn=collate_dgl,
            **cfg["data_loader_args"],
        )

    if checkpoint_dir != "":
        os.makedirs(checkpoint_dir, exist_ok=True)

    if metrics_path != "":
        Path(metrics_path).mkdir(exist_ok=True, parents=True)

    all_retrains_metrics = {}

    for retrain_idx in trange(cfg["num_retrains"], desc="Retrains"):
        model = get_model(model_name, model_args=cfg["args"], device=device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, **cfg["step_lr"])

        writer = None
        if log_dir != "":
            writer = SummaryWriter(
                log_dir=os.path.join(log_dir, str(retrain_idx))
            )

        epochs = cfg["learning_args"]["epochs"]
        reg = src.utils.get_module_from_str(cfg["reg"])()

        chkt_dir = Path(checkpoint_dir).joinpath(str(retrain_idx))
        chkt_dir.mkdir(exist_ok=True, parents=True)

        st_dir = Path(save_test_dir).joinpath(str(retrain_idx))
        st_dir.mkdir(exist_ok=True, parents=True)

        metrics = trainer(
            model=model,
            model_name=model_name,
            train_fn=pyg_train if not use_dgl else dgl_train,
            eval_fn=pyg_eval if not use_dgl else dgl_eval,
            test_fn=pyg_test if not use_dgl else dgl_test,
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
            checkpoint_dir=chkt_dir,
            save_test_dir=st_dir,
        )

        all_retrains_metrics[retrain_idx] = metrics

        if metrics_path != "":
            mp = Path(metrics_path).joinpath(f"{retrain_idx}.json")
            with open(mp, "w+") as f:
                json.dump(obj=metrics, fp=f, indent=4)

    if metrics_path != "":
        mp = Path(metrics_path).joinpath("all.json")
        with open(mp, "w+") as f:
            json.dump(obj=all_retrains_metrics, fp=f, indent=4)


if __name__ == "__main__":
    typer.run(main)
