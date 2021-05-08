import os
from typing import Dict, Union

import numpy as np
from ogb.lsc import PygPCQM4MDataset
from ogb.utils import smiles2graph

from src import DATA_DIR


def custom_smiles2graph(smiles: str) -> Dict[str, Union[np.ndarray, int]]:
    # We should improve the initial feature extraction
    return smiles2graph(smiles_string=smiles)


def load_dataset():
    return PygPCQM4MDataset(
        root=os.path.join(DATA_DIR, "dataset"),
        smiles2graph=custom_smiles2graph,
    )

