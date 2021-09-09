import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from ogb.utils.features import allowable_features
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.dataset import PygPCQM4MDataset, load_dataset


def load_error_groups(
    path: str = "data/predictions/error_groups.pkl",
    sample_size: int = -1,
) -> pd.DataFrame:
    with open(path, "rb") as fin:
        error_groups = pickle.load(fin)

    # Retrieve original molecule IDs
    # validation indexes should be shifted by number of training instances
    num_training_instances = np.max([
        group.max() for group in error_groups["train"].values()
    ])

    error_groups["valid"] = {
        k: v + num_training_instances
        for k, v in error_groups["valid"].items()
    }

    # Convert to DataFrame
    records = []

    for split in ("train", "valid"):
        for group_id, molecule_ids in error_groups[split].items():
            if sample_size != -1:
                # Sample subset of molecules
                molecule_ids = molecule_ids[
                    np.random.choice(molecule_ids.shape[0], size=sample_size)
                ]

            for molecule_id in molecule_ids.tolist():
                records.append({
                    "molecule_id": molecule_id,
                    "error_group": group_id,
                })

    df = pd.DataFrame.from_records(records).set_index("molecule_id")

    return df


def compute_fingerprint(data: Data) -> torch.Tensor:
    """Counts the number of distincs atoms in a molecule."""
    num_distinct_atoms = len(allowable_features["possible_atomic_num_list"])

    fingerprint = torch.zeros(size=(num_distinct_atoms,))

    for idx in data.x[:, 0].tolist():
        fingerprint[idx] += 1

    return fingerprint


def main():
    error_groups = load_error_groups(sample_size=100)
    dataset = load_dataset(loader=PygPCQM4MDataset)

    # Compute fingerprints for all molecules
    fingerprints = torch.stack([
        compute_fingerprint(data=dataset[molecule_id])
        for molecule_id in tqdm(error_groups.index)
    ], dim=0)

    fingerprints_2d = PCA(n_components=2).fit_transform(fingerprints)
    labels = error_groups["error_group"]

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.scatterplot(
        x=fingerprints_2d[:, 0],
        y=fingerprints_2d[:, 1],
        hue=labels,
        ax=ax,
    )

    output_path = "data/predictions/fingerprints/node_types.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)


if __name__ == "__main__":
    main()
