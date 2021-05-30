from argparse import Namespace
import json
import os
import random
import sys
from typing import List

import numpy as np
from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator
import torch
from torch import nn
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

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()

        _, ae_loss = model(batch)

        ae_loss.backward()
        optimizer.step()

        ae_loss_accum += ae_loss.detach().cpu().item()

    return ae_loss_accum / len(loader)


def eval(
    graph_model: nn.Module,
    hlgp_model: "HLGapPredictor",
    loader: DataLoader,
    evaluator: PCQM4MEvaluator,
) -> float:
    graph_model.eval()

    y_true = torch.cat([
        batch.y.cpu()
        for batch in tqdm(loader, desc="Eval - batch")
    ], dim=0)

    y_pred = hlgp_model.predict(graph_model=graph_model, loader=loader)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    mae = evaluator.eval(input_dict)["mae"]

    return mae


def test(
    graph_model: nn.Module,
    hlgp_model: "HLGapPredictor",
    loader: DataLoader,
):
    graph_model.eval()
    y_pred = hlgp_model.predict(graph_model=graph_model, loader=loader)

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


class HLGapPredictor:

    def __init__(
        self,
        emb_dim: int,
        epochs: int,
        device: torch.device,
        lr: float = 1e-3,
    ):
        self.device = device

        self.epochs = epochs

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 1),
        )

        self.optimizer = torch.optim.AdamW(
            params=self.layers.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )
        self.loss_fn = nn.L1Loss()

        self.layers.to(self.device)

    def fit(self, graph_model: nn.Module, loader: DataLoader) -> List[float]:
        graph_model.eval()
        self.layers.train()

        losses = []

        for _ in tqdm(range(self.epochs), desc="HLGapPredictor - epochs"):
            total_loss = 0

            for batch in tqdm(
                iterable=loader,
                desc="HLGapPredictor - batch",
                leave=False,
            ):
                batch = batch.to(self.device)

                with torch.no_grad():
                    z_graph = graph_model(batch)

                self.optimizer.zero_grad()

                hlg_pred = self.layers(z_graph).squeeze(dim=-1)
                hlg_true = batch.y

                loss = self.loss_fn(input=hlg_pred, target=hlg_true)

                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            losses.append(total_loss / len(loader))

        return losses

    def predict(
        self,
        graph_model: nn.Module,
        loader: DataLoader
    ) -> torch.Tensor:
        graph_model.eval()
        self.layers.train()

        hl_pred = []

        for batch in tqdm(loader, desc="Predict HLGapPredictor"):
            batch = batch.to(self.device)

            with torch.no_grad():
                z_graph = graph_model(batch)
                hlp = (
                    self.layers(z_graph)
                    .cpu()
                    .squeeze(dim=-1)
                    .clamp(min=0, max=50)
                )

                hl_pred.append(hlp)

        return torch.cat(hl_pred, dim=0)


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
        train_ae_loss = train(model, device, train_loader, optimizer)

        hl_gap_predictor = HLGapPredictor(
            emb_dim=params.emb_dim,
            epochs=5,
            device=device,
        )
        hlgp_loss = hl_gap_predictor.fit(graph_model=model, loader=train_loader)

        print('Evaluating...')
        train_mae = eval(
            graph_model=model,
            hlgp_model=hl_gap_predictor,
            loader=train_loader,
            evaluator=evaluator,
        )
        valid_mae = eval(
            graph_model=model,
            hlgp_model=hl_gap_predictor,
            loader=valid_loader,
            evaluator=evaluator,
        )

        print({
            "AE loss": train_ae_loss,
            "HLGapPredictor loss": hlgp_loss[-1],
            "Train MAE": train_mae,
            "Validation MAE": valid_mae,
        })

        if params.log_dir != '':
            writer.add_scalar("loss/model", train_ae_loss, epoch)

            for idx in range(hl_gap_predictor.epochs):
                writer.add_scalar(
                    "loss/hlgp",
                    hlgp_loss[idx],
                    epoch * hl_gap_predictor.epochs + idx
                )

            writer.add_scalar("mae/train", train_mae, epoch)
            writer.add_scalar("mae/valid", valid_mae, epoch)

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
                y_pred = test(
                    graph_model=model,
                    hlgp_model=hl_gap_predictor,
                    loader=test_loader,
                )
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
