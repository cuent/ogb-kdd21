import importlib
from typing import Any

import torch
import torch as th
from torch.nn import Identity


def batch2tensor(batch_adj, batch_feat, node_per_pool_graph):
    """
    transform a batched graph to batched adjacency tensor and node feature tensor
    """
    batch_size = int(batch_adj.size()[0] / node_per_pool_graph)
    adj_list = []
    feat_list = []
    for i in range(batch_size):
        start = i * node_per_pool_graph
        end = (i + 1) * node_per_pool_graph
        adj_list.append(batch_adj[start:end, start:end])
        feat_list.append(batch_feat[start:end, :])
    adj_list = list(map(lambda x: th.unsqueeze(x, 0), adj_list))
    feat_list = list(map(lambda x: th.unsqueeze(x, 0), feat_list))
    adj = th.cat(adj_list, dim=0)
    feat = th.cat(feat_list, dim=0)

    return feat, adj


def get_module_from_str(module: str) -> Any:
    module, cls = module.rsplit(".", maxsplit=1)
    cls = getattr(importlib.import_module(module), cls)
    return cls


def move_to(obj, device):
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list) or isinstance(obj, tuple):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError(type(obj), "Invalid type for move_to")


def load_model(
    model,
    checkpoint_path,
    test_dataloader,
    evaluator,
    eval_fn,
    device,
):
    # init bn
    model(
        move_to(next(iter(test_dataloader)), device)
    )

    state_dict = torch.load(checkpoint_path)["model_state_dict"]
    model.load_state_dict(state_dict)

    print(
        "Loaded model score",
        eval_fn(
            model=model,
            loader=test_dataloader,
            evaluator=evaluator,
            device=device,
        ),
    )

    model.graph_pred_linear = Identity()
    return model
