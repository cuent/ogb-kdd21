import os
import pickle
import random

import numpy as np
import torch
import typer
import yaml

from src.dataset import load_dataset
from src.defaults import DATASETS, MODELS
from src.model_utils import get_models

app = typer.Typer()


def setup_seed() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)


def main(
    config_path: str = typer.Option(..., help="config path"),
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
    train_dataloaders = {}
    valid_dataloaders = {}
    test_dataloaders = {}
    model_datasets = {}
    for model in cfg["models"]:
        model_name = list(model.keys())[0]

        model_type = model[model_name]["model"]
        model_ds = DATASETS[model_type]["name"]

        model_datasets[model_type] = model_ds
        if model_ds not in datasets:
            datasets[model_ds] = load_dataset(
                loader=DATASETS[model_type]["cls"]
            )

            (
                train_dataloaders[model_ds],
                valid_dataloaders[model_ds],
                test_dataloaders[model_ds],
            ) = DATASETS[model_type]["loader_fn"](
                dataset=datasets[model_ds],
                split_idx=split_idx,
                batch_size=256,
                num_workers=num_workers,
                shuffle_train=False,
                shuffle_test=False,
                shuffle_valid=False,
            )

    models, model_types_mapping = get_models(
        cfg,
        valid_dataloaders,
        device=device,
        ignore_pred_layer=False,
        datasets=DATASETS,
        models_cls=MODELS,
    )
    predictions = {}
    for model_name, model in models.items():
        model_type = model_types_mapping[model_name]
        model_ds = DATASETS[model_type]["name"]

        y_tr_pred, y_tr_true = DATASETS[model_type]["test_fn"](
            model=model, device=device, loader=train_dataloaders[model_ds]
        )
        y_ts_pred, y_ts_true = DATASETS[model_type]["test_fn"](
            model=model, device=device, loader=test_dataloaders[model_ds]
        )
        y_val_pred, y_val_true = DATASETS[model_type]["test_fn"](
            model=model, device=device, loader=valid_dataloaders[model_ds]
        )
        predictions[model_name] = {
            "train": {"y_true": y_tr_true, "y_pred": y_tr_pred},
            "valid": {"y_true": y_val_true, "y_pred": y_val_pred},
            "test": {"y_true": y_ts_true, "y_pred": y_ts_pred},
        }

    with open("data/predictions/predictions.pkl", "wb") as f:
        pickle.dump(obj=predictions, file=f)


if __name__ == "__main__":
    typer.run(main)
