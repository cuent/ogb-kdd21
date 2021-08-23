import torch
import torch_geometric
from tqdm.auto import tqdm

from src.utils import move_to


def pyg_train(model, device, loader, optimizer, gnn_name, loss_fn):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if isinstance(batch, torch_geometric.data.batch.Batch):
            batch = batch.to(device)
            y = batch.y
        else:
            batch = move_to(obj=batch, device=device)
            y = batch["y"]

        pred = model(batch).view(-1)

        optimizer.zero_grad()
        loss = loss_fn(pred, y)

        if gnn_name == "gin-virtual-bnn":
            kl_loss = model.get_kl_loss()[0]
            loss += kl_loss

        loss.backward()
        optimizer.step()


def pyg_eval(model, device, loader, evaluator, loss_fn):
    model.eval()

    loss_acc = 0
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if isinstance(batch, torch_geometric.data.batch.Batch):
            batch = batch.to(device)
            y = batch.y
        else:
            batch = move_to(obj=batch, device=device)
            y = batch["y"]

        with torch.no_grad():
            pred = model(batch).view(-1)

        loss_acc += loss_fn(pred, y).detach().cpu().item()
        y_true.append(y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    return {
        "loss": loss_acc / (step + 1),
        "mae": evaluator.eval({
            "y_true": torch.cat(y_true, dim=0),
            "y_pred": torch.cat(y_pred, dim=0),
        })["mae"],
    }


def pyg_test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if isinstance(batch, torch_geometric.data.batch.Batch):
            batch = batch.to(device)
        else:
            batch = move_to(obj=batch, device=device)

        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    return y_pred
