import gzip
import os
from pathlib import Path
from typing import Callable, Optional

import dgl
import numpy as np
import pandas as pd
import torch
from ogb.lsc import PygPCQM4MDataset, DglPCQM4MDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm

from src import DATA_DIR
from src.converters import smiles2graphft


def load_dataset(
    loader: Callable = PygPCQM4MDataset,
    smiles2graph_fn: Optional[Callable] = None,
):
    return loader(
        root=os.path.join(DATA_DIR, "dataset"),
        smiles2graph=smiles2graph_fn,
    )


def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)

    return batched_graph, labels


def get_dgl_dataloaders(
    dataset: DglPCQM4MDataset,
    batch_size: int,
    num_workers: int,
):
    split_idx = dataset.get_idx_split()
    split_idx["train"] = split_idx["train"].type(torch.LongTensor)
    split_idx["test"] = split_idx["test"].type(torch.LongTensor)
    split_idx["valid"] = split_idx["valid"].type(torch.LongTensor)

    train_loader = torch.utils.data.DataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_dgl,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset[split_idx["valid"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_dgl,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset[split_idx["test"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_dgl,
    )

    return train_loader, valid_loader, test_loader


def get_data_loaders(
    dataset: PygPCQM4MDataset,
    split_idx: dict,
    batch_size: int,
    num_workers: int,
    save_test_dir: str,
    train_subset: bool = False,
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


class CustomPCQM4MDataset:
    def __init__(
        self,
        output_path: str = "data/dataset/pcqm4m_kddcup2021/processed/graph_ft.pt",
        raw_data_path: str = "data/dataset/pcqm4m_kddcup2021/raw/data.csv.gz",
    ):
        self.data = None
        self.path = Path(output_path)
        self.raw_path = Path(raw_data_path)

        if not self.path.exists():
            self.process()

        self.load()

    def __getitem__(self, idx):
        return self.data[idx]

    def load(self):
        self.data = torch.load(self.path)

    def process(self):
        with gzip.open(self.raw_path, "rb") as f:
            raw_data = pd.read_csv(f)

        smiles_str = raw_data["smiles"]
        hg = raw_data["homolumogap"].values

        processed = []

        for it in tqdm(smiles_str):
            processed.append(smiles2graphft(it))

        processed = torch.from_numpy(np.array(processed))
        torch.save(obj=(processed, hg), f=self.path)
