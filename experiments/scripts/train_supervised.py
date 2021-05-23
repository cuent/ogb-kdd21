import os
import random
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
from src.dataset import load_dataset, get_data_loaders
from src.models.bayesian_gnn import BayesianGNN
from src.models.diffpool import DiffPoolGNN
from src.models.gnn import GNN
from src.training import train, eval, test


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
    model: str = typer.Option(..., help="Model name"),
    config_path: str = typer.Option(..., help="Config path"),
    device: int = typer.Option(0, help="which gpu to use if any (default: 0)"),
    num_workers: int = typer.Option(0, help="number of workers (default: 0)"),
    log_dir: str = typer.Option("", help="tensorboard log directory"),
    checkpoint_dir: str = typer.Option("", help="directory to save checkpoint"),
    save_test_dir: str = typer.Option(
        "", help="directory to save test submission file"
    ),
):
    # Training settings
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    setup_seed()
    device = torch.device(
        "cuda:" + str(device) if torch.cuda.is_available() else "cpu"
    )

    # Automatic dataloading and splitting
    dataset = load_dataset()
    split_idx = dataset.get_idx_split()

    # automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    train_loader, valid_loader, test_loader = get_data_loaders(
        dataset=dataset,
        split_idx=split_idx,
        num_workers=num_workers,
        save_test_dir=save_test_dir,
        **cfg["data_loader_args"],
    )

    if checkpoint_dir != "":
        os.makedirs(checkpoint_dir, exist_ok=True)

    model = get_model(model, model_args=cfg["args"], device=device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, **cfg["step_lr"])

    writer = None
    if log_dir != "":
        writer = SummaryWriter(log_dir=log_dir)

    best_valid_mae = 1000

    epochs = cfg["learning_args"]["epochs"]
    reg = src.get_module_from_str(cfg["reg"])()
    for epoch in range(1, epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_mae = train(
            model,
            device,
            train_loader,
            optimizer,
            gnn_name=model,
            reg_criterion=reg,
        )

        print("Evaluating...")
        valid_mae = eval(model, device, valid_loader, evaluator)

        print({"Train": train_mae, "Validation": valid_mae})

        if writer:
            writer.add_scalar("valid/mae", valid_mae, epoch)
            writer.add_scalar("train/mae", train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if checkpoint_dir != "":
                print("Saving checkpoint...")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mae": best_valid_mae,
                    "num_params": num_params,
                }
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, "checkpoint.pt"),
                )

            if save_test_dir != "":
                print("Predicting on test data...")
                y_pred = test(model, device, test_loader)
                print("Saving test submission file...")
                evaluator.save_test_submission(
                    {"y_pred": y_pred}, save_test_dir
                )

        scheduler.step()
        print(f"Best validation MAE so far: {best_valid_mae}")

    if writer:
        writer.close()


if __name__ == "__main__":
    typer.run(main)
