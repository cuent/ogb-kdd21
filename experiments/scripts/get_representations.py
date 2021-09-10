import os
import pathlib
import pickle
import random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
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


def get_split_idxs_from_error_groups(
    items_to_sample: int = 1000,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    out_dict = {}
    idx_to_group = {}
    data = pd.read_pickle("data/predictions/error_groups.pkl")
    for split_key, split_groups in data.items():

        split_group_items = []
        idx_to_group[split_key] = {}
        for group_id, (group_name, group_items) in enumerate(
            split_groups.items()
        ):
            sampled_items = np.random.choice(
                group_items, size=items_to_sample, replace=False
            )
            assert len(set(sampled_items)) == items_to_sample
            split_group_items.extend(sampled_items)
            idx_to_group[split_key].update(
                dict(
                    zip(
                        np.arange(
                            group_id * items_to_sample,
                            (group_id + 1) * items_to_sample,
                        ),
                        group_name * items_to_sample,
                    )
                )
            )
        out_dict[split_key] = np.array(split_group_items)
    return out_dict, idx_to_group


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

    split_idx, idx_to_group = get_split_idxs_from_error_groups()
    pathlib.Path("data/predictions").mkdir(parents=True, exist_ok=True)
    with open("data/predictions/error_group_split_idx.pkl", "wb") as f:
        pickle.dump(obj=split_idx, file=f)
    with open(
        "data/predictions/error_group_split_idx_groups_mapping.pkl", "wb"
    ) as f:
        pickle.dump(obj=idx_to_group, file=f)

    split_idx["test"] = np.array([0])
    split_idx = {
        k: torch.from_numpy(v).to(dtype=torch.long)
        for k, v in split_idx.items()
    }

    datasets = {}
    train_dataloaders = {}
    valid_dataloaders = {}
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
                _,
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
        ignore_pred_layer=True,
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
        y_val_pred, y_val_true = DATASETS[model_type]["test_fn"](
            model=model, device=device, loader=valid_dataloaders[model_ds]
        )
        predictions[model_name] = {
            "train": y_tr_pred,
            "valid": y_val_pred,
        }

    output_path = Path("data/predictions/representations.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(obj=predictions, file=f)


if __name__ == "__main__":
    typer.run(main)
