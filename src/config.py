from __future__ import annotations
from pydantic import (
    BaseModel,
    field_validator,
    validator,
    PositiveInt,
    DirectoryPath,
    FilePath,
)
from typing import Literal, List
import yaml
from typing import Optional, Union
from pathlib import Path
import torch


class Config(BaseModel):
    metadata_path: FilePath  # downloaded from https://www.rxrx.ai/rxrx1 : Metadata
    images_dir: DirectoryPath
    model: dict = {"type": "densenet121"}
    num_epochs: int = 2
    learning_rate: float = 1e-4
    train_batch_size: int = 32
    test_batch_size: int = 12
    loss_ce_weight: float = 0.8
    num_categories: int = 15
    resize_img_dim: int = 224
    cell_embedding_dim: int = 12
    data_augmentation: list[str] = (
        ["vertical", "horizontal", "rotate", "cutmix", "crop:224"],
    )
    arcface_loss: dict = {"s": 30, "m": 0.5}
    wandb: Optional[dict] = None
    save_dir: DirectoryPath = "./model1_checkpoints/"
    # save_model_version: Union[list, str] = ["best", "last"]
    scheduler: Optional[dict] = None

    @validator("loss_ce_weight")
    def check_loss_coeff(cls, v):
        if 0 <= v <= 1:
            return v
        else:
            raise ValueError(
                "loss coefficient balances metric and cross-entropy loss. must be between 0-1"
            )

    @validator("model")
    def check_model_type(cls, v):
        """
        `model` in config.yml should look something like:
        model:
            type: denset161 or vit_b_16

        If custom vit is desired, kwargs is needed:
        model:
            type: vit
            kwargs:
                image_size: 224
                patch_size: 16
                num_heads: 4
                num_layers: 4
                hidden_dim: 100
                mlp_dim: 1028
        """
        model_type = v["type"].lower()
        if model_type.startswith("densenet"):
            assert model_type in [
                "densenet121",
                "densenet169",
                "densenet201",
                "densenet161",
            ]
        elif model_type == "vit":
            assert (
                "kwargs" in v
            ), "ViT from scratch requires kwargs to construct pytorch VisionTransformer()"

        return v

    @validator("num_categories")
    def check_num_cat(cls, v):
        if v <= 1139:
            return v
        else:
            raise ValueError("max num_categories is 1139")

    @validator("data_augmentation")
    def validate_data_augmentation(cls, v):
        recognized_augmentations = [
            "vertical",
            "horizontal",
            "rotate",
            "cutmix",
        ]
        for aug in v:
            assert aug in recognized_augmentations or aug.startswith(
                "crop:"
            ), f"Unrecognized data_augmentation: {aug}"
        return v

    @validator("scheduler")
    def validate_scheduler(cls, v):
        """
        Scheduler in config.yml should look something like:
        scheduler:
            type: CosineAnnealingWarmRestarts
            kwargs:
                T_0: 10

        where `type` is in torch.optim.lr_scheduler
        where `kwargs` are additional kwargs needed to init scheduler (except optimizer)
        """
        if v is None:
            return

        try:
            getattr(torch.optim.lr_scheduler, v["type"])
        except Exception as e:
            raise ValueError(
                f"Failed to import valid scheduler: {v['type']} from torch.optim.lr_scheduler"
            ) from e

        assert (
            "type" in v
        ), "scheduler should have the key: type, and value from  Schedulers enum"
        assert (
            "kwargs" in v
        ), "scheduler should have the key: kwargs, needed to init scheduler"
        return v

    @validator("wandb")
    def validate_wandb(cls, v):
        if v is None:
            return

        assert "project" in v, "`project` field needed for wandb project name"
        assert "name" in v, "`name` field needed for wandb experiment name"
        return v

    @validator("images_dir")
    def validate_images_dir(cls, v):
        if Path(v).is_dir():
            return v
        else:
            raise FileNotFoundError(f"images_dir does not exist: {v}")

    @validator("metadata_path")
    def validate_metadata_path(cls, v):
        if Path(v).exists():
            return v
        else:
            raise FileNotFoundError(f"metadata_path does not exist: {v}")

    @validator("save_dir")
    def validate_save_dir(cls, v):
        v = Path(v)
        v.mkdir(exist_ok=True)
        return v

    @property
    def use_cutmix(self) -> bool:
        return "cutmix" in self.data_augmentation

    @property
    def use_wandb(self) -> bool:
        return self.wandb is not None

    @property
    def use_scheduler(self) -> bool:
        return self.scheduler is not None

    @classmethod
    def load_config(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    @staticmethod
    def write_yaml(config: Config, yaml_path: str):
        pathlib_fields = ["images_dir", "metadata_path", "save_dir"]
        _config = dict(config)
        for key in pathlib_fields:
            _config[key] = str(config[key])

        with open(yaml_path, "w") as yaml_file:
            yaml.dump(_config, yaml_file)

    # @validator("save_model_version")
    # def validate_save_model_version(
    #     cls, v: Union[str, list]
    # ) -> List[Literal["best", "last", "all"]]:
    #     """
    #     Save model at:
    #         all: every epoch
    #         best: epoch with best test set accuracy
    #         last: last epoch trained
    #     """
    #     if isinstance(v, str):
    #         v = [v]

    #     for item in v:
    #         assert item in [
    #             "all",
    #             "best",  # based on test set
    #             "last",
    #         ], "Accepted `save_model_version` args: all, best, last"

    #     return v
