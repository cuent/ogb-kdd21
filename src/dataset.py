import os
from typing import Callable, Optional

import torch
from ogb.lsc import PygPCQM4MDataset
from torch_geometric.data import DataLoader

from src import DATA_DIR


def load_dataset(smiles2graph_fn: Optional[Callable] = None):
    return PygPCQM4MDataset(
        root=os.path.join(DATA_DIR, "dataset"),
        smiles2graph=smiles2graph_fn,
    )


def get_data_loaders(
    dataset: PygPCQM4MDataset,
    split_idx: dict,
    batch_size: int,
    num_workers: int,
    train_subset: bool,
    save_test_dir: str,
):
    loader_kws = dict(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[
            : int(subset_ratio * len(split_idx["train"]))
        ]
        train_idx = split_idx["train"][subset_idx]
    else:
        train_idx = split_idx["train"]

    train_loader = DataLoader(
        dataset=dataset[train_idx],
        shuffle=True,
        **loader_kws,
    )
    valid_loader = DataLoader(
        dataset=dataset[split_idx["valid"]],
        shuffle=False,
        **loader_kws,
    )

    if save_test_dir != "":
        test_loader = DataLoader(
            dataset=dataset[split_idx["test"]],
            shuffle=False,
            **loader_kws,
        )
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader
