import os
from multiprocessing import Pool
from typing import List, Optional, Tuple

import numpy as np
import time
import pytorch_lightning as pl
import skimage
import torch
from dataclasses import dataclass
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from MAPS.NoisyImmunolabeling.data import Sample


class FineTuneDataset(Dataset):
    def __init__(
        self,
        file_list: List[Sample],
        training_size: Tuple[int] = (32, 256, 256),
        data_stride: Tuple[int] = (16, 128, 128),
        mode: str = "train",
        extract_channel: Optional[int] = None,
        path_preds="",
    ):
        coords = [f.file[f.file.find("_nhs") :] for f in file_list]
        coords = [c[5:-4].split("_") for c in coords]
        self.coords = [tuple(map(int, c)) for c in coords]
        self.imgs_path_list = [os.path.join(f.path_nhs, f.get_nhs_name()) for f in file_list]
        self.targets_path_list = [os.path.join(f.path_pred, f.file.replace("nhs", "pred_mod")) for f in file_list]
        self.training_size = training_size
        self.data_stride = data_stride
        self.mode = mode
        assert self.mode in ["train", "val", "test"], f"Unvalid mode parameter: {mode}"

        self.img_list = []
        self.target_list = []

        self.ids = []  # List of extracted patches
        self.nhs_quantiles = [None] * len(file_list)
        self.nhs_mins = [None] * len(file_list)

        print(f"{time.strftime('%H:%M:%S')} Reading in NHS")

        for file_id in range(0, len(self.imgs_path_list)):
            # Read files
            imgs_path = self.imgs_path_list[file_id]
            print(f"{time.strftime('%H:%M:%S')} Processing: {imgs_path}")
            target_path = self.targets_path_list[file_id]
            cur_image = skimage.io.imread(imgs_path)
            # cur_image = nhs_images[imgs_path].copy()
            cur_coord = self.coords[file_id]

            # When we load in the full data, we often want to extract a specific channel (mostly NHS, i.e. channel 2)
            if extract_channel is not None:
                channel_dim = np.argmin(cur_image.shape)  # We assume that the channel dimension is the smallest
                cur_image = np.take(cur_image, indices=extract_channel, axis=channel_dim)

            # Load the target if its present (won't be if we use the BasicDataset for inference)
            self.target_set = skimage.io.imread(target_path).astype(
                "uint8"
            )  # /255 # Empty masks get loaded as uint16 for some reason

            # Preprocess the data and normalise the data -> 0-1
            # For the training mode, we precompute the cut-off quantiles (acts as augmentation)
            # Otherwise, if no threshold config file is given which specifies a specific max value,
            # we compute the 99% quantile and set this to 1
            # Specifically for the 2nd batch we need to give hand-picked thresholds as the data looks very different
            if self.mode == "train":
                self.upper_quantiles = [0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975, 0.999]
                self.lower_quantiles = [0.0001, 0.001, 0.0025, 0.005, 0.006, 0.0075]
                all_quantiles = np.quantile(cur_image, q=self.lower_quantiles + self.upper_quantiles)
                self.nhs_quantiles[file_id] = all_quantiles[len(self.lower_quantiles) :]
                self.nhs_mins[file_id] = all_quantiles[: len(self.lower_quantiles)]
            else:
                min_nhs, max_nhs = np.quantile(cur_image, q=[0.005, 0.995])
                cur_image = np.clip(cur_image, min_nhs, max_nhs)
                cur_image = (cur_image - min_nhs) / (max_nhs - min_nhs)

            z, y, x = cur_coord
            d, w, h = self.target_set.shape
            cur_image = cur_image[z : z + d, y : y + w, x : x + h]

            self.img_list.append(torch.tensor(cur_image.astype(np.float32), dtype=torch.float))
            self.target_list.append(torch.tensor(self.target_set, dtype=torch.float))

            end_point = cur_image.shape
            start_point = (0, 0, 0)
            # Find the positions of each extracted patch
            for i in range(start_point[0], end_point[0] - self.training_size[0] + 1, self.data_stride[0]):
                for x in range(start_point[1], end_point[1] - self.training_size[1] + 1, self.data_stride[1]):
                    for y in range(start_point[2], end_point[2] - self.training_size[2] + 1, self.data_stride[2]):
                        self.ids.append(
                            [file_id, [i, i + training_size[0]], [x, x + training_size[1]], [y, y + training_size[2]]]
                        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # Extract the patch coordinates
        file_id, z_range, x_range, y_range = self.ids[i]

        # Extract image sets
        img = self.img_list[file_id][z_range[0] : z_range[1], x_range[0] : x_range[1], y_range[0] : y_range[1]].clone()
        target = self.target_list[file_id][
            z_range[0] : z_range[1], x_range[0] : x_range[1], y_range[0] : y_range[1]
        ].clone()

        if self.mode == "train":
            # Flipping - data augmentation
            flip_data = np.random.rand(3) > 0.5
            for dim in range(3):
                if flip_data[dim]:
                    img = torch.flip(img, (dim,))
                    target = torch.flip(target, (dim,))

            # Thresholding augmentation
            # Select random quantile
            quantile_id = np.random.randint(0, len(self.nhs_quantiles[file_id]))
            quantile = self.nhs_quantiles[file_id][quantile_id]
            quantile_id = np.random.randint(0, len(self.nhs_mins[file_id]))
            cur_min = self.nhs_mins[file_id][quantile_id]

            img = (img.clip(cur_min, quantile) - cur_min) / (quantile - cur_min)

        img = FineTuneDataset.rearrange_shape(img)
        target = FineTuneDataset.rearrange_shape_target(target)

        return {"image": img, "target": target}

    @staticmethod
    def rearrange_shape(img_trans):
        if len(img_trans.shape) == 3:
            img_trans = img_trans[..., None]
        # HWC to CHW
        img_trans = rearrange(img_trans, "Z X Y C-> C Z X Y")

        return img_trans

    @staticmethod
    def add_gaussian_noise(img, std: float = 0.0025):
        noise = std * torch.randn(*img.shape)
        return img + noise

    @staticmethod
    def rearrange_shape_target(target):
        if len(target.shape) == 3:
            target = target[..., None]

        # HWC to CHW
        target = rearrange(target, "Z X Y C-> C Z X Y")
        return target


class FineTuneCellData(pl.LightningDataModule):
    def __init__(
        self,
        train_files,
        test_files,
        batch_size,
        training_size: Tuple[int] = (32, 256, 256),
        data_stride: Tuple[int] = (16, 128, 128),
        extract_channel: Optional[int] = 1,
        path_preds: str = "",
    ):
        super().__init__()
        self.train_files = train_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.training_size = training_size
        self.data_stride = data_stride
        self.extract_channel = extract_channel

        self.train_data = FineTuneDataset(
            self.train_files,
            training_size=self.training_size,
            data_stride=self.data_stride,
            mode="train",
            extract_channel=self.extract_channel,
            path_preds=path_preds,
        )
        if self.test_files:
            self.test_data = FineTuneDataset(
                self.test_files,
                training_size=self.training_size,
                data_stride=self.data_stride,
                mode="test",
                extract_channel=self.extract_channel,
                path_preds=path_preds,
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, shuffle=True, num_workers=4)
