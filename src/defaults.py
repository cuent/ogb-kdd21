from ogb.lsc import DglPCQM4MDataset, PygPCQM4MDataset

from src.dataset import (
    LinearPCQM4MDataset,
    get_dgl_data_loaders,
    get_tg_data_loaders,
)
from src.model_utils import (
    get_diffpool_model,
    get_gin_virtual_bnn_model,
    get_gin_virtual_model,
)
from src.models import LinearModel
from src.training.dgl_training import dgl_eval, dgl_test, dgl_get_representations
from src.training.pyg import pyg_eval, pyg_test, pyg_get_representations

MODELS = {
    "diffpool": get_diffpool_model,
    "linear": LinearModel,
    "gin-virtual": get_gin_virtual_model,
    "gin-virtual-bnn": get_gin_virtual_bnn_model,
}
DATASETS = {
    "diffpool": {
        "name": "dgl",
        "cls": DglPCQM4MDataset,
        "eval_fn": dgl_eval,
        "test_fn": dgl_test,
        "loader_fn": get_dgl_data_loaders,
        "representations_fn": dgl_get_representations
    },
    "gin-virtual": {
        "name": "pyg",
        "cls": PygPCQM4MDataset,
        "eval_fn": pyg_eval,
        "test_fn": pyg_test,
        "loader_fn": get_tg_data_loaders,
        "representations_fn": pyg_get_representations
    },
    "gin-virtual-bnn": {
        "name": "pyg",
        "cls": PygPCQM4MDataset,
        "eval_fn": pyg_eval,
        "test_fn": pyg_test,
        "loader_fn": get_tg_data_loaders,
        "representations_fn": pyg_get_representations
    },
    "linear": {"name": "linear", "cls": LinearPCQM4MDataset},
}
