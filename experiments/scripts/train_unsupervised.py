from argparse import Namespace
import json
import os
import random
import sys

import numpy as np
from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import yaml

from src import DATA_DIR
from src.dataset import load_dataset
from src.models.gae import GraphAE


torch.multiprocessing.set_sharing_strategy('file_system')


def train(model, device, loader, optimizer):
    model.train()
    ae_loss_accum = 0
    hl_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()

        z_graph, homo_lumo_gap, ae_loss, hl_loss = model(batch)

        ae_loss.backward()
        hl_loss.backward()
        optimizer.step()

        ae_loss_accum += ae_loss.detach().cpu().item()
        hl_loss_accum += hl_loss.detach().cpu().item()

    return (
        ae_loss_accum / len(loader),
        hl_loss_accum / len(loader),
    )


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            _, homo_lumo_gap = model(batch)

        y_true.append(batch.y.detach().cpu())
        y_pred.append(homo_lumo_gap.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            _, homo_lumo_gap = model(batch)

        y_pred.append(homo_lumo_gap.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


def get_params():
    model_name = sys.argv[1]

    with open("params.yaml", "r") as fin:
        params = yaml.safe_load(fin)[model_name]

    train_subset = "TRAIN_SUBSET" in os.environ.keys()
    epochs = params.pop("epochs")

    if train_subset:
        epochs = 1_000

    return Namespace(**{
        **params,
        "epochs": epochs,
        "model_name": model_name,
        "device": os.getenv("DEVICE", "0"),
        "train_subset": train_subset,
        "num_workers": int(os.getenv("NUM_WORKERS", 0)),
        "log_dir": os.path.join(DATA_DIR, "logs", model_name),
        "checkpoint_dir": os.path.join(DATA_DIR, "checkpoints", model_name),
        "save_test_dir": os.path.join(DATA_DIR, "submissions", model_name),
        "metrics_file": os.path.join(
            DATA_DIR, "metrics", f"{model_name}.json"
        ),
    })


def get_data_loaders(
    dataset: PygPCQM4MDataset, split_idx: dict,
    params: Namespace,
):
    loader_kws = dict(
        batch_size=params.batch_size,
        num_workers=params.num_workers,
    )

    if params.train_subset:
        subset_ratio = 0.1
        subset_idx = (
            torch.randperm(len(split_idx["train"]))
            [:int(subset_ratio * len(split_idx["train"]))]
        )
        train_idx = split_idx["train"][subset_idx]
    else:
        train_idx = split_idx["train"]

    train_loader = DataLoader(
        dataset=dataset[train_idx],
        shuffle=True,
        **loader_kws,
    )
    valid_loader = DataLoader(
        dataset=dataset[split_idx["valid"]],
        shuffle=False,
        **loader_kws,
    )

    if params.save_test_dir != '':
        test_loader = DataLoader(
            dataset=dataset[split_idx["test"]],
            shuffle=False,
            **loader_kws,
        )
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader


def get_model(params, device):
    shared_params = {
        'num_layers': params.num_layers,
        'emb_dim': params.emb_dim,
    }
    if params.model_name == 'gae':
        model = GraphAE(**shared_params)
    else:
        raise ValueError('Invalid GNN type')

    model = model.to(device)
    return model


def main():
    # Training settings
    params = get_params()

    print(params)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device(
        "cuda:" + str(params.device) if torch.cuda.is_available()
        else "cpu"
    )

    # Automatic dataloading and splitting
    dataset = load_dataset()
    split_idx = dataset.get_idx_split()

    # automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    train_loader, valid_loader, test_loader = get_data_loaders(
        dataset=dataset,
        split_idx=split_idx,
        params=params,
    )

    if params.checkpoint_dir != '':
        os.makedirs(params.checkpoint_dir, exist_ok=True)

    os.makedirs(os.path.dirname(params.metrics_file), exist_ok=True)

    model = get_model(params, device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if params.log_dir != '':
        writer = SummaryWriter(log_dir=params.log_dir)

    best_valid_mae = 1000

    if params.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    for epoch in range(1, params.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_ae_loss, train_mae = train(model, device, train_loader, optimizer)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader, evaluator)

        print({
            'AE': train_ae_loss,
            'Train': train_mae,
            'Validation': valid_mae,
        })

        if params.log_dir != '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)
            writer.add_scalar('train/ae_loss', train_ae_loss, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if params.checkpoint_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(),
                              'best_val_mae': best_valid_mae,
                              'num_params': num_params}
                torch.save(
                    obj=checkpoint,
                    f=os.path.join(params.checkpoint_dir, 'checkpoint.pt')
                )

            if params.save_test_dir != '':
                print('Predicting on test data...')
                y_pred = test(model, device, test_loader)
                print('Saving test submission file...')
                evaluator.save_test_submission(
                    input_dict={'y_pred': y_pred},
                    dir_path=params.save_test_dir,
                )

            # Save metrics
            with open(params.metrics_file, "w") as fout:
                json.dump(obj={"val_mae": best_valid_mae}, fp=fout, indent=4)

        scheduler.step()

        print(f'Best validation MAE so far: {best_valid_mae}')

    if params.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main()
