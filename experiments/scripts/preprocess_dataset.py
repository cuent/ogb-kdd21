import typer
import yaml

from src.dataset import load_dataset
from src.utils import get_module_from_str
from ogb.lsc import PygPCQM4MDataset, DglPCQM4MDataset
from src.dataset import LinearPCQM4MDataset

app = typer.Typer()

DATASETS = {
    "DglPCQM4MDataset": DglPCQM4MDataset,
    "PygPCQM4MDataset": PygPCQM4MDataset,
    "LinearPCQM4MDataset": LinearPCQM4MDataset,
}


def main(dataset: str = typer.Option(..., help="Dataset path")):
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)["preprocess_dataset"]

    s2g_converter = get_module_from_str(cfg["smiles2graph"]["cls"])(
        **cfg["smiles2graph"]["args"]
    )

    load_dataset(DATASETS[dataset], smiles2graph_fn=s2g_converter)


if __name__ == "__main__":
    typer.run(main)
