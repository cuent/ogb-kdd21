import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import typer
import yaml
from ogb.lsc import PCQM4MEvaluator
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from src.dataset import (
    AggregateCollater,
    DatasetAggregator,
    get_torch_dataloaders,
    get_y,
    load_dataset_with_validloader,
)
from src.defaults import DATASETS, MODELS
from src.model_utils import get_models
from src.models import AggregatedModel
from src.training.pyg import pyg_eval, pyg_test, pyg_train
from src.training.trainer import trainer

app = typer.Typer()


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

    models, _ = get_models(
        cfg,
        valid_dataloaders,
        device=device,
        ignore_pred_layer=True,
        datasets=DATASETS,
        models_cls=MODELS,
    )

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
