import os
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import skimage
import torch
import json
from typing import Union
from einops import rearrange
from skimage.morphology import isotropic_dilation
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

PATH_ORG = '/well/rittscher/projects/PanVision/data/FullStacks/Originals/MouseKidney_April2024'
# PATH_ORG = '/well/rittscher/projects/PanVision/data/FullStacks/Originals/MouseKidney_May2024'
PATH_PRED = '/well/rittscher/users/jyo949/tmp/KidneyTest/MajorityVote/'
PATH_PRED_OUTLINE = '/well/rittscher/users/jyo949/AntiBodySegKidney/results/combined/'


@dataclass
class Sample:
    file: str
    path_nhs: str
    path_pred_outline: str
    path_pred_inner: str


@dataclass
class FileRecord:
    file_name: str
    path_base: str
    file_outline_pred: str
    path_outline_pred: str
    file_inner_pred: str
    path_inner_pred: str
    nhs_lower: Union[int, float]
    nhs_upper: Union[int, float]


def read_data_config(dataset_config) -> List[FileRecord]:
    with open(dataset_config, 'r') as file:
        data = json.load(file)

    records = []
    for record in data:
        records.append(
            FileRecord(
                file_name=record['file_name'],
                path_base=record['path_base'],
                file_outline_pred=record['file_outline_pred'],
                path_outline_pred=record['path_outline_pred'],
                file_inner_pred=record['file_inner_pred'],
                path_inner_pred=record['path_inner_pred'],
            )
        )
    return records


class InitAdoptDataset(Dataset):
    def __init__(
        self,
        file_list: List[FileRecord],
        #  targets_path_list: Optional[List[str]] = None,
        training_size: Tuple[int] = (32, 256, 256),
        data_stride: Tuple[int] = (16, 128, 128),
        mode: str = 'train',
        extract_channel: Optional[int] = None,
        outline_dilation: int = 14,
    ):
        # self.imgs_path_list = [os.path.join(file.path_nhs, file.file) for file in file_list]
        # self.targets_path_list = [os.path.join(file.path_pred_inner, file.file) for file in file_list]
        # self.targets_outline_path_list = [
        #     os.path.join(file.path_pred_outline, file.file.replace('.tif', 'pred_bin.tif')) for file in file_list
        # ]
        self.training_size = training_size
        self.data_stride = data_stride
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], f'Unvalid mode parameter: {mode}'

        self.img_list = []
        self.target_list = []

        self.ids = []  # List of extracted patches
        self.nhs_quantiles = [None] * len(file_list)
        self.nhs_mins = [None] * len(file_list)

        for file_id in range(0, len(file_list)):
            # Read files
            file_record = file_list[file_id]
            imgs_path = os.path.join(file_record.path_base, file_record.file_name)
            target_path = os.path.join(file_record.path_inner_pred, file_record.file_inner_pred)
            outline_path = os.path.join(file_record.path_outline_pred, file_record.file_outline_pred)

            cur_image = skimage.io.imread(imgs_path)

            # When we load in the full data, we often want to extract a specific channel (mostly NHS, i.e. channel 2)
            if extract_channel is not None:
                channel_dim = np.argmin(cur_image.shape)  # We assume that the channel dimension is the smallest
                cur_image = np.take(cur_image, indices=extract_channel, axis=channel_dim)

            # Load the target if its present (won't be if we use the BasicDataset for inference)
            try:
                self.target_set = skimage.io.imread(target_path).astype(
                    'uint8'
                )  # /255 # Empty masks get loaded as uint16 for some reason
                outline = skimage.io.imread(outline_path)
                if outline_dilation > 0:
                    outline = np.stack([isotropic_dilation(s, outline_dilation) for s in outline], axis=0)
                self.target_set[outline == 0] = 0
            except FileNotFoundError:
                print(f'No file found for {target_path}')
                self.target_set = np.zeros_like(cur_image).astype('float')

            # Preprocess the data and normalise the data -> 0-1
            # For the training mode, we precompute the cut-off quantiles (acts as augmentation)
            # Otherwise, if no threshold config file is given which specifies a specific max value,
            # we compute the 99% quantile and set this to 1
            # Specifically for the 2nd batch we need to give hand-picked thresholds as the data looks very different
            if self.mode == 'train':
                self.upper_quantiles = [0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975, 0.999]
                self.lower_quantiles = [0.0001, 0.001, 0.0025, 0.005, 0.006, 0.0076]
                all_quantiles = np.quantile(cur_image, q=self.lower_quantiles + self.upper_quantiles)
                self.nhs_quantiles[file_id] = all_quantiles[len(self.lower_quantiles) :]
                self.nhs_mins[file_id] = all_quantiles[: len(self.lower_quantiles)]
            else:
                min_nhs, max_nhs = np.quantile(cur_image, q=[0.005, 0.995])
                cur_image = np.clip(cur_image, min_nhs, max_nhs)
                cur_image = (cur_image - min_nhs) / (max_nhs - min_nhs)

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

        if self.mode == 'train':
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

        img = InitAdoptDataset.rearrange_shape(img)
        target = InitAdoptDataset.rearrange_shape_target(target)

        return {'image': img, 'target': target}

    @staticmethod
    def rearrange_shape(img_trans):
        if len(img_trans.shape) == 3:
            img_trans = img_trans[..., None]
        # HWC to CHW
        img_trans = rearrange(img_trans, 'Z X Y C-> C Z X Y')

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
        target = rearrange(target, 'Z X Y C-> C Z X Y')
        return target


class KidneyPredictDataset(Dataset):
    def __init__(
        self,
        imgs_path_list: List[str],
        #  targets_path_list: Optional[List[str]] = None,
        training_size: Tuple[int] = (32, 256, 256),
        data_stride: Tuple[int] = (16, 128, 128),
        mode: str = 'train',
        extract_channel: Optional[int] = None,
    ):
        self.imgs_path_list = imgs_path_list
        self.training_size = training_size
        self.data_stride = data_stride

        self.img_list = []
        self.target_list = []
        self.mode = mode

        self.ids = []  # List of extracted patches
        self.nhs_quantiles = [None] * len(imgs_path_list)
        self.nhs_mins = [None] * len(imgs_path_list)

        for file_id in range(0, len(self.imgs_path_list)):
            # Read files
            imgs_path = self.imgs_path_list[file_id]

            try:
                cur_image = skimage.io.imread(imgs_path)
            except FileNotFoundError:
                cur_image = skimage.io.imread(imgs_path.replace('April', 'May'))

            # When we load in the full data, we often want to extract a specific channel (mostly NHS, i.e. channel 2)
            if extract_channel is not None:
                channel_dim = np.argmin(cur_image.shape)  # We assume that the channel dimension is the smallest
                cur_image = np.take(cur_image, indices=extract_channel, axis=channel_dim)

            # Preprocess the data and normalise the data -> 0-1
            # For the training mode, we precompute the cut-off quantiles (acts as augmentation)
            # Otherwise, if no threshold config file is given which specifies a specific max value,
            # we compute the 99% quantile and set this to 1
            # Specifically for the 2nd batch we need to give hand-picked thresholds as the data looks very different
            if self.mode == 'train':
                self.upper_quantiles = [0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975, 0.999]
                self.lower_quantiles = [0.0001, 0.001, 0.0025, 0.005, 0.006, 0.0076]
                all_quantiles = np.quantile(cur_image, q=self.lower_quantiles + self.upper_quantiles)
                self.nhs_quantiles[file_id] = all_quantiles[len(self.lower_quantiles) :]
                self.nhs_mins[file_id] = all_quantiles[: len(self.lower_quantiles)]
            else:
                min_nhs, max_nhs = np.quantile(cur_image, q=[0.005, 0.995])
                cur_image = np.clip(cur_image, min_nhs, max_nhs)
                cur_image = (cur_image - min_nhs) / (max_nhs - min_nhs)

            self.img_list.append(torch.tensor(cur_image.astype(np.float32), dtype=torch.float))

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

        if self.mode == 'train':
            # Flipping - data augmentation
            flip_data = np.random.rand(3) > 0.5
            for dim in range(3):
                if flip_data[dim]:
                    img = torch.flip(img, (dim,))

            # Thresholding augmentation
            # Select random quantile
            quantile_id = np.random.randint(0, len(self.nhs_quantiles[file_id]))
            quantile = self.nhs_quantiles[file_id][quantile_id]
            quantile_id = np.random.randint(0, len(self.nhs_mins[file_id]))
            cur_min = self.nhs_mins[file_id][quantile_id]

            img = (img.clip(cur_min, quantile) - cur_min) / (quantile - cur_min)

        img = InitAdoptDataset.rearrange_shape(img)

        return {'image': img}

    @staticmethod
    def rearrange_shape(img_trans):
        if len(img_trans.shape) == 3:
            img_trans = img_trans[..., None]
        # HWC to CHW
        img_trans = rearrange(img_trans, 'Z X Y C-> C Z X Y')

        return img_trans

    @staticmethod
    def add_gaussian_noise(img, std: float = 0.0025):
        noise = std * torch.randn(*img.shape)
        return img + noise


class CellData(pl.LightningDataModule):
    def __init__(
        self,
        train_files,
        test_files,
        batch_size,
        training_size: Tuple[int] = (32, 256, 256),
        data_stride: Tuple[int] = (16, 128, 128),
        extract_channel: Optional[int] = 1,
        outline_dilation: int = 14,
    ):
        super().__init__()
        self.train_files = train_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.training_size = training_size
        self.data_stride = data_stride
        self.extract_channel = extract_channel

        self.train_data = InitAdoptDataset(
            self.train_files,
            training_size=self.training_size,
            data_stride=self.data_stride,
            mode='train',
            extract_channel=self.extract_channel,
            outline_dilation=outline_dilation,
        )
        if self.test_files:
            self.test_data = InitAdoptDataset(
                self.test_files,
                training_size=self.training_size,
                data_stride=self.data_stride,
                mode='test',
                extract_channel=self.extract_channel,
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_data, self.batch_size)
