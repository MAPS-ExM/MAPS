import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange
import skimage
import tifffile
import contextlib
import json
from typing import List, Tuple, Union, Optional


class BasicDataset(Dataset):
    def __init__(
        self,
        imgs_dir: List[str],
        target_dir: Optional[List[str]] = None,
        training_size: Tuple[int] = (32, 256, 256),
        data_stride: Tuple[int] = (16, 128, 128),
        scale: float = 1.0,
        start_point: Tuple[int] = (0, 0, 0),
        end_point: Tuple[int] = [200, 2048, 2048],
        only_data_w_target_present: bool = False,
        background_threshold: float = 0.0,
        mode: str = "train",
        extract_channel: Optional[int] = None,
        config_file_threshold: Optional[str] = None,
        nhs_lower_threshold_quantile: Optional[float] = None,
    ):
        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        self.imgs_dir_list = imgs_dir
        self.target_dir_list = target_dir if target_dir is not None else [""] * len(imgs_dir)
        self.training_size = training_size
        self.data_stride = data_stride
        self.end_point_original = end_point.copy()
        self.scale = scale
        self.background_threshold = background_threshold
        self.mode = mode
        assert self.mode in ["train", "val", "test"], f"Unvalid mode parameter: {mode}"

        self.img_list = []
        self.target_list = []
        self.only_data_w_target_present = only_data_w_target_present

        self.ids = []  # List of extracted patches
        self.nhs_quantiles = [None] * len(imgs_dir)
        self.nhs_mins = [None] * len(imgs_dir)

        # Configuration file which gives hand picked NHS max and min values
        # Especially needed for the 2nd batch
        if config_file_threshold is not None:
            with open(config_file_threshold, "r") as f:
                self.threshold_config = json.load(f)

        # Save drug/ condition of each file
        self.drug_mapping = {"None": 0, "HBSS": 1, "Antimycin": 2, "Oligomycin": 3, "CCCP": 4}
        self.drug_list = torch.Tensor(self.extract_drug_information(self.imgs_dir_list)).long()

        for file_id in range(0, len(self.imgs_dir_list)):
            # Read files
            imgs_path = self.imgs_dir_list[file_id]
            target_path = self.target_dir_list[file_id]
            cur_image = skimage.io.imread(imgs_path)

            # if cur_image.ndim == 3:
            #     cur_image = cur_image[..., None]

            # If the data is too small, mirror it to increase the number of slices
            if cur_image.shape[0] < training_size[0]:
                cur_z = cur_image.shape[0]
                print(f"Warning: Mirror {imgs_path} to increase slices from {cur_z} to {training_size[0]}")
                diff = training_size[0] - cur_z

                extra_slices = cur_image[cur_z - 1 - diff : -1]
                cur_image = np.concatenate((cur_image, extra_slices[::-1]), axis=0)
                assert cur_image.shape[0] == training_size[0], (
                    f"Error: {imgs_path} has {cur_image.shape[0]} slices, but {training_size[0]} for training specified"
                )

            # When we load in the full data, we often want to extract a specific channel (mostly NHS, i.e. channel 2)
            if extract_channel is not None:
                channel_dim = np.argmin(cur_image.shape)  # We assume that the channel dimension is the smallest
                cur_image = np.take(cur_image, indices=extract_channel, axis=channel_dim)

            if target_path != "":
                # Load the target if its present (won't be if we use the BasicDataset for inference)
                self.target_set = skimage.io.imread(target_path).astype(
                    "uint8"
                )  # /255 # Empty masks get loaded as uint16 for some reason
                if self.target_set.shape[0] < training_size[0]:
                    extra_slices = self.target_set[cur_z - 1 - diff : -1]
                    self.target_set = np.concatenate((self.target_set, extra_slices[::-1]), axis=0)
            else:
                print("No target given")
                self.target_set = np.zeros(cur_image.shape)

            # Preprocess the data and normalise the data -> 0-1
            # For the training mode, we precompute the cut-off quantiles (acts as augmentation)
            # Otherwise, if no threshold config file is given which specifies a specific max value,
            # we compute the 99% quantile and set this to 1
            # Specifically for the 2nd batch we need to give hand-picked thresholds as the data looks very different
            if self.mode == "train":
                self.nhs_quantiles[file_id] = np.quantile(cur_image, q=[0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975])
                self.nhs_mins[file_id] = np.min(cur_image)
            elif config_file_threshold is None or imgs_path not in self.threshold_config:
                if nhs_lower_threshold_quantile is not None:
                    min_nhs, max_nhs = np.quantile(cur_image, q=[nhs_lower_threshold_quantile, 0.99])
                else:
                    max_nhs, min_nhs = np.quantile(cur_image, q=0.99), np.min(cur_image)
                # cur_image[cur_image > max_nhs] = max_nhs
                cur_image = np.clip(cur_image, min_nhs, max_nhs)
                cur_image = (cur_image - min_nhs) / (max_nhs - min_nhs)
            else:
                thresholds = self.threshold_config[imgs_path]["nhs"]
                cmin, cmax, left_offset = thresholds["min"], thresholds["max"], thresholds["left_offset"]
                cur_image = np.clip(cur_image, cmin, cmax)
                cur_image = (cur_image - cmin) / (cmax - cmin)

                if left_offset != 0:
                    cur_image = (cur_image + left_offset * (1 + left_offset)) / (1 + left_offset)

            if self.scale != 1:
                # Upsample the image and mask if the scale is !=1
                scale_factor = (1, int(1 / self.scale), int(1 / self.scale))
                cur_image = skimage.transform.downscale_local_mean(cur_image, scale_factor, cval=0, clip=True)
                if len(self.target_set.shape) == 4:
                    scale_factor = (1, int(1 / scale), int(1 / scale), 1)
                else:
                    scale_factor = (1, int(1 / scale), int(1 / scale))
                self.target_set = skimage.transform.downscale_local_mean(
                    self.target_set, scale_factor, cval=0, clip=True
                )

            self.img_list.append(torch.tensor(cur_image.astype(np.float32), dtype=torch.float))
            self.target_list.append(torch.tensor(self.target_set, dtype=torch.float))

            end_point[0] = np.min((self.end_point_original[0], cur_image.shape[0]))
            end_point[1] = np.min((self.end_point_original[1], cur_image.shape[1]))
            end_point[2] = np.min((self.end_point_original[2], cur_image.shape[2]))

            # Find the positions of each extracted patch
            for i in range(start_point[0], end_point[0] - self.training_size[0] + 1, self.data_stride[0]):
                for x in range(start_point[1], end_point[1] - self.training_size[1] + 1, self.data_stride[1]):
                    for y in range(start_point[2], end_point[2] - self.training_size[2] + 1, self.data_stride[2]):
                        if self.only_data_w_target_present:
                            # If selected, only keep patches which have a least a bit of target in them
                            if len(self.target_set.shape) == 4:
                                target_present = (
                                    np.sum(
                                        self.target_set[
                                            i : i + training_size[0],
                                            x : x + training_size[1],
                                            y : y + training_size[2],
                                            :,
                                        ]
                                    )
                                    > 0
                                )
                            elif len(self.target_set.shape) == 3:
                                target_present = (
                                    np.sum(
                                        self.target_set[
                                            i : i + training_size[0], x : x + training_size[1], y : y + training_size[2]
                                        ]
                                    )
                                    > 0
                                )
                            else:
                                ValueError("Unexpected target size")
                        else:
                            target_present = True

                        # In order to cut down on the number of background patches, we can set a threshold
                        # for the mean of the patch. If the mean is below the threshold, we don't include it
                        if self.background_threshold and self.background_threshold != 1:
                            nhs_mean = np.mean(
                                cur_image[i : i + training_size[0], x : x + training_size[1], y : y + training_size[2]]
                            )

                            if nhs_mean < self.background_threshold:
                                continue  # Don't include this area

                        if target_present:
                            self.ids.append(
                                [
                                    file_id,
                                    [i, i + training_size[0]],
                                    [x, x + training_size[1]],
                                    [y, y + training_size[2]],
                                ]
                            )

    def extract_drug_information(self, file_list: List[str]) -> List[str]:
        """Scans each file name for a drug name and saves it, if none are found, 'None' is saved instead"""
        drugs = []
        for file in file_list:
            curr_drug = "None"
            if "HBSS" in file:
                curr_drug = "HBSS"
            elif "Antimycin" in file:
                curr_drug = "Antimycin"
            elif "Oligomycin" in file:
                curr_drug = "Oligomycin"
            elif "CCCP" in file:
                curr_drug = "CCCP"
            drugs.append(self.drug_mapping[curr_drug])
        return drugs

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
        drug_treament = self.drug_list[file_id]

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
            cur_min = self.nhs_mins[file_id]

            img = (img.clip(cur_min, quantile) - cur_min) / (quantile - cur_min)
            # img = BasicDataset.add_gaussian_noise(img).clip(0,1)
        elif self.mode == "pretrain_val":
            # In the pre-trainig phase we don't use a separate dataset but just pick a
            # set of random patches.
            # As the Dataset gets initialised with mode=='train' and the mode is only changed to 'pretrain_val'
            # after the random split, we still need to normalise the patches when we retrieve them.
            # We always choose the center quantile to do so.
            quantile_id = len(self.nhs_quantiles[file_id]) // 2 - 1
            quantile = self.nhs_quantiles[file_id][quantile_id]
            cur_min = self.nhs_mins[file_id]
            img = (img.clip(cur_min, quantile) - cur_min) / (quantile - cur_min)

        img = BasicDataset.rearrange_shape(img)
        target = BasicDataset.rearrange_shape_target(target)

        return {"image": img, "target": target, "drug_treatment": drug_treament}

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


class BinaryBasicDataset(BasicDataset):
    """For a binary target, we replace the target with 0/1 values and turn it into a float tensor"""

    def __init__(
        self,
        imgs_dir,
        target_dir,
        training_size=(32, 128, 128),
        data_stride=(1, 1, 1),
        scale=1,
        start_point=(0, 0, 0),
        end_point=(200, 2048, 2048),
        only_data_w_target_present=False,
        background_threshold=None,
    ):
        super().__init__(
            imgs_dir,
            target_dir,
            training_size,
            data_stride,
            scale,
            start_point,
            end_point,
            only_data_w_target_present,
            background_threshold,
        )

        # Make targets binary and turn into float
        for i in range(len(self.target_list)):
            cur_target = self.target_list[i]
            cur_target[cur_target > 0] = 1
            self.target_list[i] = cur_target.float()


class MultiClassBasicDataset(BasicDataset):
    """For a multi-class target, we need to transform the target into a long tensor"""

    def __getitem__(self, item):
        data = super().__getitem__(item)
        data["target"] = data["target"].long()
        return data


class MultiClassFineTuningDataset(BasicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # For the fine-tuning setting, we replace the slices which are not annotated with -1 which is the ignore index in the loss functions
        for i in range(len(self.target_list)):
            cur_target = self.target_list[i].long()
            empty_slices = torch.sum(torch.abs(cur_target), dim=(1, 2)) == 0
            cur_target[empty_slices] = -1

            self.target_list[i] = cur_target
