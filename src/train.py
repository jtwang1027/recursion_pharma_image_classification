import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import wandb
from torchvision.transforms import v2
from functools import partial
import dill as pickle  # needed for collate_fn
import torch.nn.functional as F
import argparse
from typing import Literal


from dataset import Rxrx1, make_transform_pipeline
from config import Config
from models import CustomDensenet, CustomVit
from losses import ArcFaceLoss, calc_accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))


def load_model(
    model_type="densenet121",
    num_categories: int = 15,
    cell_embedding_dim: int = 12,
    image_size: int = 224,
):
    """Convenience wrapper to load any model type."""

    if "densenet" in model_type:
        # densenet121, densenet169, densenet201, densenet161
        model = CustomDensenet(
            backbone=model_type,
            num_classes=num_categories,
            cell_embedding_dim=cell_embedding_dim,
        )
    else:
        # ViT
        model = CustomVit(
            image_size=image_size,
            num_classes=num_categories,
            dropout=0.2,
            cell_embedding_dim=cell_embedding_dim,
        )

    return model


def cutmix_collate_fn(batch_list: list, num_categories: int):
    # each batch contains: x (img tensor), cell_type (int), label (int)
    cutmix = v2.CutMix(num_classes=num_categories)

    x_stack = torch.stack([item[0] for item in batch_list], dim=0)
    cell_type_stack = torch.stack([torch.tensor(item[1]) for item in batch_list], dim=0)
    label_stack = torch.stack([torch.tensor(item[2]) for item in batch_list], dim=0)

    x_stack, label_stack = cutmix(x_stack, label_stack)
    return x_stack, cell_type_stack, label_stack


@torch.no_grad
def eval(model, test_dataloader):
    """Evaluate model on entire test set."""
    all_cftn_predictions = []
    all_metric_head_predictions = []
    all_labels = []

    for x, cell_type, labels in test_dataloader:
        x, cell_type, labels = x.to(device), cell_type.to(device), labels.to(device)
        metric_pred, cftn_pred = model(x, cell_type)
        all_cftn_predictions.append(cftn_pred)
        all_metric_head_predictions.append(metric_pred)
        all_labels.append(labels)

    labels_tensor = torch.cat(all_labels)
    cftn_head_accuracy = calc_accuracy(torch.cat(all_cftn_predictions), labels_tensor)
    metric_head_accuracy = calc_accuracy(
        torch.cat(all_metric_head_predictions), labels_tensor
    )

    return metric_head_accuracy, cftn_head_accuracy


def setup_dataloader(config: Config, split: Literal["train", "test"]):
    """Create Rxrx1 pytorch dataset and dataloader for train or test split."""

    if config.use_cutmix and split == "train":
        collate_fn = partial(cutmix_collate_fn, num_categories=config.num_categories)
    else:
        collate_fn = None

    if split == "train":
        transform_pipeline = make_transform_pipeline(
            resize_dim=config.resize_img_dim, transform_list=config.data_augmentation
        )
    else:  # `test` split
        transform_pipeline = make_transform_pipeline(
            resize_dim=config.resize_img_dim, transform_list=[]
        )

    print(f"Setting up {split} dataset and dataloader.")
    rxrx1_dataset = Rxrx1(
        images_dir=config.images_dir,
        metadata_path=config.metadata_path,
        split=split,
        num_categories=config.num_categories,
        data_transform=transform_pipeline,
    )

    rxrx1_dataloader = DataLoader(
        rxrx1_dataset,
        batch_size=getattr(config, f"{split}_batch_size"),
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )
    return rxrx1_dataset, rxrx1_dataloader


def train(config: Config):
    learning_rate = config.learning_rate
    num_categories = config.num_categories
    num_epochs = config.num_epochs
    use_wandb = config.use_wandb
    loss_ce_weight = config.loss_ce_weight

    # setup train/test datasets/dataloaders
    _, train_dataloader = setup_dataloader(config, "train")
    _, test_dataloader = setup_dataloader(config, "test")

    # use a single batch for test set
    test_x, test_cell_type, test_labels = next(iter(test_dataloader))
    test_x, test_cell_type, test_labels = (
        test_x.to(device),
        test_cell_type.to(device),
        test_labels.to(device),
    )

    if config.use_wandb:
        wandb.init(
            project=config.wandb["project"],
            name=config.wandb["name"],
            config=dict(config),
        )

    print("Loading model.")
    model = load_model(
        config.model, num_categories=num_categories, image_size=config.resize_img_dim
    )
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    metric_loss = ArcFaceLoss(**config.arcface_loss)
    ce_loss = nn.CrossEntropyLoss()

    if config.use_scheduler:
        scheduler = getattr(torch.optim.lr_scheduler, config.scheduler["type"])(
            optimizer, **config.scheduler["kwargs"]
        )

    print("Starting training.")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")

        for batch_num, (x, cell_type, labels) in tqdm(enumerate(train_dataloader)):
            x = x.to(device)
            cell_type = cell_type.to(device)
            labels = labels.to(device)

            # predictions for: metric loss and classification loss
            pred_embedding_metric, pred_cftn = model(x, cell_type)

            if config.use_cutmix:
                _metric_loss = metric_loss(
                    pred_embedding_metric, labels
                )  # when using cutmix, OHE not needed
            else:
                _metric_loss = metric_loss(
                    pred_embedding_metric,
                    F.one_hot(labels, num_classes=config.num_categories),
                )

            _ce_loss = ce_loss(pred_cftn, labels)
            loss = loss_ce_weight * _ce_loss + (1 - loss_ce_weight) * _metric_loss

            # train_accuracy = calc_accuracy(pred_cftn, labels) # can't calc accuracy using cutmix

            wandb_log_info = {
                "loss": loss.item(),
                "train_ce_loss": _ce_loss.item(),
                "metric_loss": _metric_loss.item(),
                "epoch": epoch,
                # 'train_accuracy':train_accuracy,
            }

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EVAL
            if batch_num % 50 == 0:
                with torch.no_grad():
                    test_metric, test_cftn = model(test_x, test_cell_type)
                    test_accuracy = calc_accuracy(test_cftn, test_labels)
                    test_metric_accuracy = calc_accuracy(
                        test_metric, test_labels
                    )  # from metric loss head
                    test_ce_loss = ce_loss(test_cftn, test_labels)

                wandb_log_info.update(
                    {
                        # test set
                        "test_accuracy": test_accuracy,
                        "test_accuracy:metric_head": test_metric_accuracy,
                        "test_ce_loss": test_ce_loss.item(),
                    }
                )

            if use_wandb:
                wandb.log(wandb_log_info)
            else:
                print(wandb_log_info)

        if config.use_scheduler:
            scheduler.step()

        torch.save(model.state_dict(), config.save_dir / f"{epoch}.pt")

    print("Training completed.")
    metric_head_accuracy, cftn_head_accuracy = eval(model, test_dataloader)

    if use_wandb:
        print("Running eval on entire test set.")
        wandb.run.summary["full_test_accuracy:cftn_head"] = cftn_head_accuracy
        wandb.run.summary["full_test_accuracy:metric_head"] = metric_head_accuracy
        wandb.finish()
    else:
        print(f"full test set accuracy (ce loss head): {cftn_head_accuracy}")
        print(f"full test set accuracy (metric loss head): {metric_head_accuracy}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="example_config.yaml",
        help="local path to config.yaml",
    )  # , required=True)

    args = parser.parse_args()
    config: Config = Config.load_config(args.config_path)
    print("Config loaded.")
    print(config)
    train(config)
