import os
import json
from multiprocessing import Pool
from typing import List, Tuple
import numpy as np
import tifffile
import torch
from skimage import morphology
from torch.utils.data import Dataset


from MAPS.NoisyImmunolabeling.data import FileRecord


def load_ab_dilated(ab, path):
    if not os.path.exists(path):
        ab_dil = np.zeros_like(ab).astype("int")
        # You might want to fine-tune these parameters to your dataset! This is $R_\tau$ in the paper
        # It really depends on what your immunolabelling looks like. This is what worked for a
        # Mix of Mic60 and CoxIV for mitochondria in mouse kidney tissue.
        ab = morphology.remove_small_objects(ab, min_size=9)
        for i in range(ab_dil.shape[0]):
            ab_dil[i] = morphology.binary_dilation(ab[i] > 0, footprint=morphology.disk(12))
            ab_dil[i] = morphology.remove_small_objects(ab_dil[i] > 0, min_size=650)
            ab_dil[i] = morphology.binary_closing(ab_dil[i], footprint=morphology.disk(10))

        # Remove single antibodies
        ab_dil = morphology.remove_small_objects(ab_dil > 0, min_size=2000)

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        tifffile.imwrite(path, ab_dil.astype("uint8"), compression="zlib")
    else:
        ab_dil = tifffile.imread(path)
    return ab_dil


class BasicDataset(Dataset):
    def __init__(
        self,
        img_records: List[FileRecord],
        training_size: Tuple[int] = (16, 256, 256),
        data_stride: Tuple[int] = (8, 128, 128),
        mode: str = "train",
    ):
        self.imgs_path_list = [os.path.join(file.path_base, file.file_name) for file in img_records]
        self.dilated_ab_list = [
            os.path.join(
                file.path_base,
                "DilatedAntiBodyMask",
                file.file_name.replace(".tif", f"_dil_{file.ab_threshold}_new.tif"),
            )
            for file in img_records
        ]
        self.training_size = training_size
        self.data_stride = data_stride
        self.mode = mode
        assert self.mode in ["train", "val", "test"], f"Unvalid mode parameter: {mode}"

        self.img_list = []
        self.ab_list = []
        self.ab_dilated_list = []

        self.ids = []  # List of extracted patches
        self.nhs_quantiles = [None] * len(self.imgs_path_list)
        self.nhs_mins = [None] * len(self.imgs_path_list)

        if len(self.imgs_path_list) > 1:
            with Pool(8) as p:
                org_images = p.map(tifffile.imread, self.imgs_path_list)
        else:
            org_images = [tifffile.imread(p) for p in self.imgs_path_list]

        for file_id in range(0, len(self.imgs_path_list)):
            # Read files
            dilated_ab_path = self.dilated_ab_list[file_id]
            cur_image = org_images[file_id]
            cur_ab = cur_image[:, 0]
            cur_image = cur_image[:, 1]

            cur_ab = cur_ab > img_records[file_id].ab_threshold

            cur_ab_dilated = load_ab_dilated(cur_ab, dilated_ab_path).astype("uint8")
            cur_ab[cur_ab_dilated == 0] = 0

            # Preprocess the data and normalise the data -> 0-1
            # For the training mode, we precompute the cut-off quantiles (acts as augmentation)
            # Otherwise, if no threshold config file is given which specifies a specific max value,
            # we compute the 99% quantile and set this to 1
            if self.mode == "train":
                self.nhs_quantiles[file_id] = np.quantile(
                    cur_image, q=[0.9875, 0.99, 0.9925, 0.994, 0.995, 0.996, 0.9975]
                )
                if img_records[file_id].nhs_lower <= 1:
                    min_nhs = np.quantile(cur_image, q=img_records[file_id].nhs_lower)
                else:
                    min_nhs = img_records[file_id].nhs_lower

                self.nhs_mins[file_id] = max(np.min(cur_image), min_nhs)
            else:
                # For inference/ testing, we modify the whole image
                if img_records[file_id].nhs_lower <= 1:
                    min_nhs = np.quantile(cur_image, q=img_records[file_id].nhs_lower)
                else:
                    min_nhs = img_records[file_id].nhs_lower

                if img_records[file_id].nhs_upper <= 1:
                    max_nhs = np.quantile(cur_image, q=img_records[file_id].nhs_upper)
                else:
                    max_nhs = img_records[file_id].nhs_upper

                cur_image = np.clip(cur_image, min_nhs, max_nhs)
                cur_image = (cur_image - min_nhs) / (max_nhs - min_nhs)

            self.img_list.append(torch.tensor(cur_image.astype(np.float32), dtype=torch.float))
            self.ab_list.append(torch.tensor(cur_ab.astype(np.float32), dtype=torch.float))
            self.ab_dilated_list.append(torch.tensor(cur_ab_dilated, dtype=torch.float))

            end_point = cur_image.shape
            start_point = (0, 0, 0)
            # Find the positions of each extracted patch
            for i in range(
                start_point[0],
                end_point[0] - self.training_size[0] + 1,
                self.data_stride[0],
            ):
                for x in range(
                    start_point[1],
                    end_point[1] - self.training_size[1] + 1,
                    self.data_stride[1],
                ):
                    for y in range(
                        start_point[2],
                        end_point[2] - self.training_size[2] + 1,
                        self.data_stride[2],
                    ):
                        self.ids.append(
                            [
                                file_id,
                                [i, i + training_size[0]],
                                [x, x + training_size[1]],
                                [y, y + training_size[2]],
                            ]
                        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # Extract the patch coordinates
        file_id, z_range, x_range, y_range = self.ids[i]

        # Extract image sets
        img = self.img_list[file_id][z_range[0] : z_range[1], x_range[0] : x_range[1], y_range[0] : y_range[1]].clone()
        ab = self.ab_list[file_id][z_range[0] : z_range[1], x_range[0] : x_range[1], y_range[0] : y_range[1]].clone()
        ab_dilated = self.ab_dilated_list[file_id][
            z_range[0] : z_range[1], x_range[0] : x_range[1], y_range[0] : y_range[1]
        ].clone()

        if self.mode == "train":
            # Flipping - data augmentation
            flip_data = np.random.rand(3) > 0.5
            for dim in range(3):
                if flip_data[dim]:
                    img = torch.flip(img, (dim,))
                    ab = torch.flip(ab, (dim,))
                    ab_dilated = torch.flip(ab_dilated, (dim,))

            # Thresholding augmentation
            # Select random quantile
            quantile_id = np.random.randint(0, len(self.nhs_quantiles[file_id]))
            quantile = self.nhs_quantiles[file_id][quantile_id]
            cur_min = self.nhs_mins[file_id]

            img = (img.clip(cur_min, quantile) - cur_min) / (quantile - cur_min)

        img = img[None]
        ab = ab[None]
        ab_dilated = ab_dilated[None]

        return img, ab_dilated, ab


def read_data_config(dataset_config) -> List[FileRecord]:
    with open(dataset_config, "r") as file:
        data = json.load(file)

    records = []
    for record in data:
        records.append(
            FileRecord(
                file_name=record["file_name"],
                path_base=record["path_base"],
                ab_threshold=record["ab_threshold"],
                nhs_lower=record.get("nhs_lower", 0.005),
                nhs_upper=record.get("nhs_upper", 0.995),
            )
        )
    return records


def build_data(dataset_config=""):
    """
    Expects path as dataset_config to a .json file specifiying the dataset with entries looking like
        {
            "file_name": "example1.tif",
            "path_base": "/path/to/files/",
            "ab_threshold": 123,
            "nhs_lower": 110,
            "nhs_upper": 0.99
        },
        {
            "file_name": "example2.tif",
            "path_base": "/path/to/another/location/",
            "ab_threshold": 124,
            "nhs_lower": 110,
            "nhs_upper": 0.99
        }

    """
    records = read_data_config(dataset_config)
    dataset = BasicDataset(
        img_records=records,
        training_size=(16, 256, 256),  # Check your GPU memory
        data_stride=(8, 128, 128),
        mode="train",
    )

    return dataset
