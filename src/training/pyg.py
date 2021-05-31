from typing import Callable

import torch
from tqdm.auto import tqdm


def pyg_train(
    model, device, loader, optimizer, gnn_name, reg_criterion: Callable
):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(batch).view(
            -1,
        )
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)

        if gnn_name == "gin-virtual-bnn":
            kl_loss = model.get_kl_loss()[0]
            loss += kl_loss

        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def pyg_eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(
                -1,
            )

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def pyg_test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(
                -1,
            )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred
