import torch
from tqdm import tqdm


def dgl_train(model, device, loader, optimizer, gnn_name, loss_fn):
    model.train()

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat").to(device)
        edge_attr = bg.edata.pop("feat").to(device)
        labels = labels.to(device)

        pred = model(bg, x, edge_attr).view(-1)

        optimizer.zero_grad()
        loss = loss_fn(pred, labels)

        loss.backward()
        optimizer.step()


def dgl_eval(model, device, loader, evaluator, loss_fn=None):
    model.eval()

    loss_acc = 0
    y_true = []
    y_pred = []

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat").to(device)
        edge_attr = bg.edata.pop("feat").to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred = model(bg, x, edge_attr).view(-1)

        if loss_fn:
            loss_acc += loss_fn(pred, labels).detach().cpu().item()
        y_true.append(labels.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    return {
        "loss": loss_acc / (step + 1),
        "mae": evaluator.eval(
            {
                "y_true": torch.cat(y_true, dim=0),
                "y_pred": torch.cat(y_pred, dim=0),
            }
        )["mae"],
    }


def dgl_test(model, device, loader):
    model.eval()
    y_pred = []
    y_true = []

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat").to(device)
        edge_attr = bg.edata.pop("feat").to(device)

        with torch.no_grad():
            pred = model(bg, x, edge_attr).view(
                -1,
            )

        y_pred.append(pred.detach().cpu())
        y_true.append(labels.view(pred.shape).detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    return y_pred, y_true


def dgl_get_representations(model, device, loader):
    model.eval()
    representations = []

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat").to(device)
        edge_attr = bg.edata.pop("feat").to(device)

        with torch.no_grad():
            pred = model(bg, x, edge_attr)

        representations.append(pred.detach().cpu())

    representations = torch.cat(representations, dim=0)

    return representations
