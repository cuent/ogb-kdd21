import yaml

from src.dataset import load_dataset
from src.utils import get_module_from_str
from ogb.lsc import PygPCQM4MDataset, DglPCQM4MDataset

def main():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)["preprocess_dataset"]

    s2g_converter = get_module_from_str(cfg["smiles2graph"]["cls"])(
        **cfg["smiles2graph"]["args"]
    )
    load_dataset(PygPCQM4MDataset, smiles2graph_fn=s2g_converter)
    load_dataset(DglPCQM4MDataset, smiles2graph_fn=s2g_converter)


if __name__ == "__main__":
    main()
