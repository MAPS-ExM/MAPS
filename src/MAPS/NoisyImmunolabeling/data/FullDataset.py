import os
import sys
from multiprocessing import Manager, Process

import numpy as np
import tifffile
import torch
from scipy.ndimage import distance_transform_cdt, value_indices
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk, isotropic_dilation
from torch.utils.data import Dataset

sys.path.append('../')
from data.Data3D import BasicDataset


def remove_holes(pred, hole_area_threshold=7500):
    for i in range(pred.shape[0]):
        labels = label(pred[i] == 0)
        regions = regionprops(labels)
        indices = value_indices(labels)
        labels_2_remove = [r.label for r in regions if r.area < hole_area_threshold]
        indices_set_2_mask = [indices[lab] for lab in labels_2_remove]

        if len(indices_set_2_mask) == 0:
            continue
        indices_set_2_mask = [
            np.concatenate([i[0] for i in indices_set_2_mask]),
            np.concatenate([i[1] for i in indices_set_2_mask]),
        ]
        pred[i][indices_set_2_mask[0], indices_set_2_mask[1]] = 1
    return pred


class SingleDataSet:
    def __init__(
        self,
        imgs_path_list,
        pred_dir,
        size,
        stride,
        mask_threshold,
        mask_pixel_reduction=0,
        mask_pixel_expansion=0,
        submask=None,
        path_base='',
        hole_size=6000,
    ):
        self.dataset = BasicDataset(
            imgs_path_list, training_size=size, data_stride=stride, mode='test', path_base=path_base
        )
        self.dataset.ab_list = None
        self.ab_dilated_list = None
        try:
            self.files_predictions = [os.path.join(pred_dir, f.replace('.tif', 'pred_bin.tif')) for f in imgs_path_list]
            self.predictions = [torch.from_numpy(tifffile.imread(f) > 0) for f in self.files_predictions]
        except FileNotFoundError:
            self.files_predictions = [os.path.join(pred_dir, f.replace('.tif', '_bin.tif')) for f in imgs_path_list]
            self.predictions = [torch.from_numpy(tifffile.imread(f) > 0) for f in self.files_predictions]
        if hole_size > 0:
            self.predictions = [remove_holes(p, hole_size) for p in self.predictions]
        self.files = [f.split('/')[-1] for f in imgs_path_list]

        for i in range(len(self.dataset.img_list)):
            assert (
                self.dataset.img_list[i].shape == self.predictions[i].shape
            ), f'Shapes do not match for {self.dataset.img_list[i]} and {self.predictions[i]}'

        if mask_pixel_reduction > 0:
            for i in range(len(self.predictions)):
                dist = distance_transform_cdt(self.predictions[i])
                self.predictions[i][dist < mask_pixel_reduction] = 0
        if mask_pixel_expansion > 0:
            for i in range(len(self.predictions)):
                for s in range(self.predictions[i].shape[0]):
                    self.predictions[i][s] = torch.from_numpy(
                        # binary_dilation(self.predictions[i][s], disk(mask_pixel_expansion))
                        isotropic_dilation(self.predictions[i][s], mask_pixel_expansion)
                    )
                # dist = distance_transform_cdt(torch.logical_not(self.predictions[i]))
                # self.predictions[i][dist < mask_pixel_reduction] = 1

        ids_with_pred = []
        for i in range(len(self.dataset)):
            file_id, z_range, x_range, y_range = self.dataset.ids[i]
            cur_pred = self.predictions[file_id][
                z_range[0] : z_range[1], x_range[0] : x_range[1], y_range[0] : y_range[1]
            ]
            # Ignore if the center pixel is not predicted
            if cur_pred[cur_pred.shape[0] // 2, cur_pred.shape[1] // 2, cur_pred.shape[2] // 2] == 0:
                continue
            if submask is not None:
                cz_range, cx_range, cy_range = submask
                cur_pred = cur_pred[cz_range[0] : cz_range[1], cx_range[0] : cx_range[1], cy_range[0] : cy_range[1]]
            if (cur_pred > 0).float().mean() > mask_threshold:
                ids_with_pred.append((file_id, z_range, x_range, y_range))

        self.dataset.ids = ids_with_pred

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        file_id, z_range, x_range, y_range = self.dataset.ids[i]

        nhs = self.dataset.img_list[file_id][
            z_range[0] : z_range[1], x_range[0] : x_range[1], y_range[0] : y_range[1]
        ].clone()
        pred = self.predictions[file_id][
            z_range[0] : z_range[1], x_range[0] : x_range[1], y_range[0] : y_range[1]
        ].clone()

        return nhs, pred


class FullDataset(Dataset):
    def __init__(
        self,
        pred_dir_wt,
        pred_dir_aki,
        size,
        stride,
        mask_threshold,
        mask_pixel_reduction=5,
        mask_pixel_expansion=0,
        submask=None,
        batch='D6',
        n_files=6,
        hole_size=0,
    ):
        path_base = '/well/rittscher/projects/PanVision/data/FullStacks/Originals/MouseKidney_April2024'
        wt_list = sorted(
            [
                f
                for f in os.listdir(path_base)
                if 'WT' in f and f.endswith('.tif') and batch in f and not f.startswith('.')
            ]
        )[:n_files]
        aki_list = sorted(
            [
                f
                for f in os.listdir(path_base)
                if 'AKI' in f and f.endswith('.tif') and batch in f and not f.startswith('.')
            ]
        )[:n_files]
        print(f'WT files: {wt_list}')
        print(f'AKI files: {aki_list}')

        # Define functions for parallel execution
        manager = Manager()
        datasets = manager.dict()

        def process_aki():
            datasets['aki_dataset'] = SingleDataSet(
                aki_list,
                pred_dir=pred_dir_aki,
                size=size,
                stride=stride,
                mask_threshold=mask_threshold,
                mask_pixel_reduction=mask_pixel_reduction,
                mask_pixel_expansion=mask_pixel_expansion,
                submask=submask,
                path_base=path_base,
                hole_size=hole_size,
            )

        def process_wt():
            datasets['wt_dataset'] = SingleDataSet(
                wt_list,
                pred_dir=pred_dir_wt,
                size=size,
                stride=stride,
                mask_threshold=mask_threshold,
                mask_pixel_expansion=mask_pixel_expansion,
                mask_pixel_reduction=5,
                submask=submask,
                path_base=path_base,
                hole_size=hole_size,
            )

        # Start processes
        p1 = Process(target=process_aki)
        p2 = Process(target=process_wt)
        p1.start()
        p2.start()

        # Wait for processes to finish
        p1.join()
        p2.join()
        # Assign datasets from the shared dictionary to attributes
        self.aki_dataset = datasets['aki_dataset']
        self.wt_dataset = datasets['wt_dataset']

        self.n_aki_dataset = len(self.aki_dataset)

    def __len__(self):
        return len(self.aki_dataset) + len(self.wt_dataset)

    def __getitem__(self, index):
        if index < self.n_aki_dataset:
            return self.aki_dataset[index]
        else:
            return self.wt_dataset[index - self.n_aki_dataset]

    def get_sample(self, i, wt=True):
        if wt:
            file_id, z_range, x_range, y_range = self.wt_dataset.dataset.ids[i]
            return self.wt_dataset[i], (
                self.wt_dataset.files[file_id],
                (z_range[0], z_range[1]),
                (x_range[0], x_range[1]),
                (y_range[0], y_range[1]),
            )  # As tuple for hash
        else:
            file_id, z_range, x_range, y_range = self.aki_dataset.dataset.ids[i]
            return self.aki_dataset[i], (
                self.aki_dataset.files[file_id],
                (z_range[0], z_range[1]),
                (x_range[0], x_range[1]),
                (y_range[0], y_range[1]),
            )
