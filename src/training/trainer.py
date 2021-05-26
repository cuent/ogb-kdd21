import os
from typing import Callable

import torch.nn
from ogb.lsc import PCQM4MEvaluator
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


def trainer(
    model: torch.nn.Module,
    model_name: str,
    train_fn: Callable,
    eval_fn: Callable,
    test_fn: Callable,
    evaluator: PCQM4MEvaluator,
    train_loader: torch.utils.data.Dataloader,
    valid_loader: torch.utils.data.Dataloader,
    test_loader: torch.utils.data.Dataloader,
    epochs: int,
    optimizer: torch.optim.optimizer.Optimizer,
    scheduler: StepLR,
    reg: Callable,
    device: torch.device,
    writer: SummaryWriter,
    checkpoint_dir: str,
    save_test_dir: str,
):
    num_params = sum(p.numel() for p in model.parameters())
    best_valid_mae = 1000

    for epoch in range(1, epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_mae = train_fn(
            model,
            device,
            train_loader,
            optimizer,
            gnn_name=model_name,
            reg_criterion=reg,
        )

        print("Evaluating...")
        valid_mae = eval_fn(model, device, valid_loader, evaluator)
        print({"Train": train_mae, "Validation": valid_mae})

        if writer:
            writer.add_scalar("valid/mae", valid_mae, epoch)
            writer.add_scalar("train/mae", train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if checkpoint_dir != "":
                print("Saving checkpoint...")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mae": best_valid_mae,
                    "num_params": num_params,
                }
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, "checkpoint.pt"),
                )

            if save_test_dir != "":
                print("Predicting on test data...")
                y_pred = test_fn(model, device, test_loader)
                print("Saving test submission file...")
                evaluator.save_test_submission(
                    {"y_pred": y_pred}, save_test_dir
                )

        scheduler.step()
        print(f"Best validation MAE so far: {best_valid_mae}")

    if writer:
        writer.close()
