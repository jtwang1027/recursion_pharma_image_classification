import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms
from PIL import Image
from typing import Optional


def make_transform_pipeline(
    resize_dim: Optional[int] = None,
    transform_list=["crop:224", "vertical", "horizontal", "rotate"],
):
    """
    Compose pytorch transform pipeline for images.
    """
    _transforms = []

    if resize_dim is not None:
        _transforms.append(transforms.Resize(resize_dim))

    if crop_dim := [
        int(i.split(":")[-1]) for i in transform_list if i.startswith("crop")
    ]:
        # apply image crop, example str format, crop:224
        crop_dim = crop_dim[0]
        _transforms.append(transforms.RandomCrop(crop_dim))  # Randomly crop the image

        if resize_dim is not None:
            assert (
                crop_dim < resize_dim
            ), f"crop dimension {crop_dim} is not less than image after resizing {resize_dim}."

    if "vertical" in transform_list:
        _transforms.append(transforms.RandomVerticalFlip())

    if "horizontal" in transform_list:
        _transforms.append(transforms.RandomVerticalFlip())

    _transforms.append(transforms.ToTensor())  # Convert the PIL Image to tensor
    return transforms.Compose(_transforms)


DEFAULT_TRANSFORM = make_transform_pipeline(
    None,
    [
        "vertical",
        "horizontal",
    ],
)


class Rxrx1(Dataset):
    def __init__(
        self,
        images_dir="/kaggle/input/recursion-cellular-image-classification/train/",
        metadata_path="/kaggle/input/rxrx1-metadata-csv/metadata.csv",  # downloaded from https://www.rxrx.ai/rxrx1 : Metadata
        split="train",
        num_categories=10,
        data_transform=DEFAULT_TRANSFORM,
    ):

        meta = pd.read_csv(metadata_path)
        meta["sirna"] = meta["sirna"].astype(
            "category"
        )  # numeric codes are assigned here
        meta["sirna_codes"] = meta["sirna"].cat.codes.astype("int64")
        meta["cell_type"] = meta["cell_type"].astype("category")
        meta["cell_type_codes"] = meta["cell_type"].cat.codes.astype(
            "int64"
        )  # int64 needed for cross entropy loss

        meta = meta[meta["dataset"] == split]

        if num_categories is not None:
            meta = meta[meta["sirna_codes"].isin(list(range(num_categories)))]
        else:
            num_categories = meta["sirna"].nunique()

        meta = meta.reset_index(drop=True)
        self.num_categories = num_categories
        self.meta = meta
        self.images_dir = Path(images_dir)

        self.transform = data_transform

    def _get_all_channel_paths(self, row: pd.Series) -> list:
        """Assemble paths for all 6 channels from metadata row"""

        full_paths = []
        for channel_num in range(1, 7):

            full_paths.append(
                self.images_dir
                / f"{row['experiment']}/Plate{row['plate']}/{row['well']}_s{row['site']}_w{channel_num}.png"
            )

            if not full_paths[-1].exists():
                raise FileNotFoundError(full_paths[-1])

        return full_paths

    def _standardize_tensor(self, x):
        # (num_channels x H x W), standardize across each channel
        # todo: check how to to this when image is cropped?
        _mean = torch.mean(x, dim=(1, 2), keepdim=True)
        _std = torch.std(x, dim=(1, 2), keepdim=True)

        return (x - _mean) / _std

    def __getitem__(self, idx):
        row: pd.Series = self.meta.iloc[idx]
        full_paths = self._get_all_channel_paths(row)
        x = torch.cat([self.transform(Image.open(p)) for p in full_paths], dim=0)
        x = self._standardize_tensor(x)

        cell_type: int = self.meta["cell_type_codes"][idx]

        label: int = self.meta["sirna_codes"][idx]

        return x, cell_type, label

    def __len__(self):
        return len(self.meta)

    @property
    def num_cell_types(self):
        return self.meta["cell_type"].nunique()

    @property
    def img_size(self):
        # get size from example
        return self[0][0].size(-1)
