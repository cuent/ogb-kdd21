from typing import Callable

import torch
from tqdm.auto import tqdm


def dgl_train(
    model, device, loader, optimizer, gnn_name, reg_criterion: Callable
):
    model.train()

    loss_accum = 0
    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat").to(device)
        edge_attr = bg.edata.pop("feat").to(device)
        labels = labels.to(device)

        pred = model(bg, x, edge_attr).view(
            -1,
        )
        optimizer.zero_grad()
        loss = reg_criterion(pred, labels)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def dgl_eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat").to(device)
        edge_attr = bg.edata.pop("feat").to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred = model(bg, x, edge_attr).view(
                -1,
            )

        y_true.append(labels.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.pyg_eval(input_dict)["mae"]


def dgl_test(model, device, loader):
    model.eval()
    y_pred = []

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat").to(device)
        edge_attr = bg.edata.pop("feat").to(device)

        with torch.no_grad():
            pred = model(bg, x, edge_attr).view(
                -1,
            )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred
