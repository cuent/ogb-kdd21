import os
from typing import Callable

import torch.nn
from ogb.lsc import PCQM4MEvaluator
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer


def trainer(
    model: torch.nn.Module,
    model_name: str,
    train_fn: Callable,
    eval_fn: Callable,
    test_fn: Callable,
    evaluator: PCQM4MEvaluator,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int,
    optimizer: Optimizer,
    scheduler: StepLR,
    reg: Callable,
    device: torch.device,
    writer: SummaryWriter,
    checkpoint_dir: str,
    save_test_dir: str,
):
    num_params = sum(p.numel() for p in model.parameters())

    train_res = None
    valid_res = None

    best_valid_res = {"mae": 1_000}
    best_valid_res_train = None

    for epoch in range(1, epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_fn(
            model=model,
            device=device,
            loader=train_loader,
            optimizer=optimizer,
            gnn_name=model_name,
            loss_fn=reg,
        )

        print("Evaluating...")
        train_res = eval_fn(model, device, train_loader, evaluator, loss_fn=reg)
        valid_res = eval_fn(model, device, valid_loader, evaluator, loss_fn=reg)

        print({"Train": train_res, "Validation": valid_res})

        if writer:
            writer.add_scalar("loss/train", train_res["loss"], epoch)
            writer.add_scalar("loss/valid", valid_res["loss"], epoch)

            writer.add_scalar("mae/train", train_res["mae"], epoch)
            writer.add_scalar("mae/valid", valid_res["mae"], epoch)

        if valid_res["mae"] < best_valid_res["mae"]:
            best_valid_res = valid_res
            best_valid_res_train = train_res

            if checkpoint_dir != "":
                print("Saving checkpoint...")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mae": best_valid_res["mae"],
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
        print(f"Best validation MAE so far: {best_valid_res['mae']}")

    if writer:
        writer.close()

    return {
        "train_res": train_res,
        "valid_res": valid_res,

        "best_valid_res": best_valid_res,
        "best_valid_res_train": best_valid_res_train,
    }
