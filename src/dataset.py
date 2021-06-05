import gzip
import os
from pathlib import Path
from typing import Callable, Optional, Any
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
from ogb.lsc import PygPCQM4MDataset, DglPCQM4MDataset
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
from torch_geometric.data.dataloader import Collater
from tqdm import tqdm

import dgl
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


def get_torch_dataloaders(
    dataset: DglPCQM4MDataset,
    split_idx: Dict[str, torch.tensor],
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
):

    train_loader = torch.utils.data.DataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset[split_idx["valid"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset[split_idx["test"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, test_loader


def get_data_loaders(
    dataset: Any,
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


class LinearPCQM4MDataset:
    def __init__(
        self,
        output_path: str = "data/dataset/pcqm4m_kddcup2021/processed/graph_ft.pt",
        raw_data_path: str = "data/dataset/pcqm4m_kddcup2021/raw/data.csv.gz",
        split_dict_path: str = "data/dataset/pcqm4m_kddcup2021/split_dict.pt",
        smiles2graph: Optional[Callable] = None,
    ):
        self.data = None
        self.path = Path(output_path)
        self.raw_path = Path(raw_data_path)
        self.split_dict_path = Path(split_dict_path)

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

        processed = self.scale(np.array(processed))
        torch.save(obj=(processed, torch.from_numpy(hg)), f=self.path)

    def scale(self, data):
        splits = torch.load(self.split_dict_path)

        splitted_data = {
            "train": {
                "data": [data[it] for it in splits["train"]],
                "split": splits["train"],
            },
            "test": {
                "data": [data[it] for it in splits["test"]],
                "split": splits["test"],
            },
            "dev": {
                "data": [data[it] for it in splits["dev"]],
                "split": splits["dev"],
            },
        }

        scaler = StandardScaler()
        splitted_data["train"]["data"] = scaler.fit_transform(
            splitted_data["train"]["data"]
        )
        splitted_data["test"]["data"] = scaler.transform(
            splitted_data["test"]["data"]
        )
        splitted_data["dev"]["data"] = scaler.transform(
            splitted_data["dev"]["data"]
        )

        out = torch.zeros(size=data.shape)

        for split, split_data in splitted_data.items():
            for it_id, ds_id in enumerate(split_data["split"]):
                out[ds_id] = torch.from_numpy(split_data["data"][it_id])

        return out


class DatasetAggregator(torch.utils.data.Dataset):
    def __init__(
        self, datasets: Dict[str, Union[torch.utils.data.Dataset, object]]
    ) -> None:
        self.datasets = datasets

    def get_ds_item(self, ds: str, idx: int):
        return self.datasets.get(ds)[idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        return {
            ds: self.get_ds_item(ds, idx)
            for ds, values in self.datasets.items()
        }

    def __len__(self):
        return len(self.datasets[list(self.datasets.keys())[0]])


class AggregateCollater:
    def __init__(self, keys):
        self.COLLATER_MAPPING = {
            "pyg": AggregateCollater.collate_pyg_agg,
            "dgl": AggregateCollater.collate_dgl_agg,
            "ft": AggregateCollater.collate_torch_agg,
            "y": AggregateCollater.collate_y,
        }
        self.keys = keys

    def __call__(self, samples):
        samples = list(samples)
        return {
            k: self.COLLATER_MAPPING[k]([it[k] for it in samples])
            for k in self.keys
        }

    @staticmethod
    def collate_y(samples):
        return torch.stack(samples)

    @staticmethod
    def collate_pyg_agg(samples):
        pyg_collater = Collater([])
        return pyg_collater(samples)

    @staticmethod
    def collate_dgl_agg(samples):
        dgl_batch = collate_dgl(samples)[:-1]
        n_features = dgl_batch.ndata.pop("feat")
        e_features = dgl_batch.edata.pop("feat")
        return (dgl_batch, n_features, e_features)

    @staticmethod
    def collate_torch_agg(samples):
        return torch.stack([it[0] for it in samples])
