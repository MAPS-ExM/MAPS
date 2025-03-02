"""
Run with subJob full_prediction.py  --memory 120 --env lightning --gpu_type 'a100-pcie-80gb'
"""

import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import skimage
import tifffile
import torch

from MAPS.MitoTracker.data import BasicDataset
from MAPS.MitoTracker.models import build_model
from MAPS.MitoTracker.utils import Args, Tracker
from MAPS.MitoTracker.utils.predict import predict_stack, remove_mirrored_slices


def load_model_from_config(model_yaml: str) -> torch.nn.Module:
    """
    Builds a model from a configuration file and loads the best model.
    """
    args = Args(base_yaml_file=model_yaml, extension_yaml_file=None)
    args.log_results = False  # Make sure we don't overwrite anything
    args.device = torch.device("cuda")

    # Initialise Tracker and set up output directories
    tracker = Tracker(
        args.path_output,
        log_results=False,
        base_config=model_yaml,
        extension_config=None,
        args=args,
    )

    # Build and load best model
    model = build_model(args)
    model = tracker.loadBestModel(model, device=args.device)
    model.eval()

    return model


def make_prediction(
    img_path: str,
    model: torch.nn.Module,
    model_yaml: str,
    patch_size: tuple = None,
    stride: tuple = None,
    extract_channel: int = None,
    threshold_file: Optional[str] = None,
    nhs_lower_threshold_quantile: Optional[float] = None,
) -> np.ndarray:
    """Take as input the path to an image file <img_path> and the model and a path to an model configuration.
    The <patch_size> and <stride> give the size of image patches that are fed into the model,
    the bigger the patches the faster is the prediction because he don't need to make too much predictions
    for the overlapping area but the more GPU memory is needed."""

    args = Args(base_yaml_file=model_yaml, extension_yaml_file=None)
    prediction_window_size = patch_size if patch_size is not None else args.training_size
    prediction_stride = stride if stride is not None else args.data_stride
    print(
        f"Using patch size: {prediction_window_size} with stride {prediction_stride}",
        flush=True,
    )

    # Make prediction
    if getattr(args, "input_channels", 1) == 1:
        dataset = BasicDataset(
            imgs_dir=[img_path],
            scale=args.scale,
            extract_channel=extract_channel,
            config_file_threshold=threshold_file,
            mode="test",
            nhs_lower_threshold_quantile=nhs_lower_threshold_quantile,
        )
        data = dataset.img_list[0]
        if getattr(args, "cat_emb_dim", None) is not None:
            drug_treatment = dataset.drug_list[0]
            prediction = predict_stack(
                data,
                model,
                drug_treatment=drug_treatment,
                img_window=prediction_window_size,
                stride=prediction_stride,
            )
        else:
            prediction = predict_stack(data, model, img_window=prediction_window_size, stride=prediction_stride)
        prediction = remove_mirrored_slices(prediction, prediction_window_size[0] // 2)
    else:
        raise NotImplementedError("Only accept one input channel")

    if args.scale != 1:
        scale_factor = np.array((1, int(1 / args.scale), int(1 / args.scale)))
        output_size = (prediction.shape * scale_factor).astype("uint16")
        prediction = skimage.transform.resize_local_mean(prediction, output_size, preserve_range=True)
    return prediction


def shorten_filename(filename: str) -> str:
    return (
        filename.replace("_TOM20647_Mitotracker_NHSester488", "")
        .replace(".ims Resolution Level 1", "")
        .replace(".ims_Resolution_Level_1", "")
    )


def get_paths(batch: int = 1, group: int = 1) -> Tuple[str, str, str]:
    if batch == 1:
        if group == 1:
            filenames = [
                "Antimycin_A_TOM20647_Mitotracker_NHSester488.ims_Resolution_Level_1.tif",
                "Antimycin_A_TOM20647_Mitotracker_NHSester488_1.ims_Resolution_Level_1.tif",
                "Antimycin_A_TOM20647_Mitotracker_NHSester488_2.ims_Resolution_Level_1.tif",
                "Antimycin_A_TOM20647_Mitotracker_NHSester488_3.ims_Resolution_Level_1.tif",
                "Antimycin_A_TOM20647_Mitotracker_NHSester488_4.ims_Resolution_Level_1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488.ims_Resolution_Level_1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_1.ims_Resolution_Level_1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_2.ims_Resolution_Level_1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_3.ims_Resolution_Level_1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_4.ims_Resolution_Level_1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488.ims_Resolution_Level_1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_1.ims_Resolution_Level_1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_2.ims_Resolution_Level_1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_3.ims_Resolution_Level_1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_4.ims_Resolution_Level_1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488.ims_Resolution_Level_1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_1.ims_Resolution_Level_1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_2.ims_Resolution_Level_1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_3.ims_Resolution_Level_1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_4.ims_Resolution_Level_1.tif",
                "Oligomycin_A_TOM20647_Mitotracker_NHSester488.ims_Resolution_Level_1.tif",
                "Oligomycin_A_TOM20647_Mitotracker_NHSester488_1.ims_Resolution_Level_1.tif",
                "Oligomycin_A_TOM20647_Mitotracker_NHSester488_2.ims_Resolution_Level_1.tif",
                "Oligomycin_A_TOM20647_Mitotracker_NHSester488_3.ims_Resolution_Level_1.tif",
                "Oligomycin_A_TOM20647_Mitotracker_NHSester488_4.ims_Resolution_Level_1.tif",
            ]
            input_file_path = f"/well/rittscher/projects/PanVision/data/FullStacks/Originals/"
            threshold_file = None
        elif group == 2:
            filenames = [
                "Antimycin A_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
                "Antimycin A_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
                "CCCP_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
                "DMSO_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
                "HBSS_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
                "Oligomycin A_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
            ]

            input_file_path = f"/well/rittscher/projects/PanVision/data/FullStacks/Originals/Drugs_2023_02_27"
            threshold_file = None
        else:
            raise ValueError(f"batch {batch} group {group} not included yet!")

    elif batch == 2:
        filenames = [
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
            "Antimycine_2nd_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif',
            # 'CCCP_2nd_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif',
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
            "DMSO_2nd_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
            "HBSS_2nd_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
            "OligomycineA_2nd_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
        ]
        input_file_path = f"/well/rittscher/projects/PanVision/data/FullStacks/Originals/Drugs_2023_02_18"
        threshold_file = "/well/rittscher/projects/PanVision/data/FullStacks/Originals/Drugs_2023_02_18/thresholds.json"

    elif batch == 3:
        filenames = [
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_15.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_16.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
            "Antimycine_3rd_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_15.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
            # "CCCP_3rd_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_15.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
            "DMSO_3rd_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_15.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
            "HBSS_3rd_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_1.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_10.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_11.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_12.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_13.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_14.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_15.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_16.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_17.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_18.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_19.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_2.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_3.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_4.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_5.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_6.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_7.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_8.ims Resolution Level 1.tif",
            "Oligomycin_3rd_TOM20647_Mitotracker_NHSester488_9.ims Resolution Level 1.tif",
        ]
        input_file_path = f"/well/rittscher/projects/PanVision/data/FullStacks/Originals/Drugs_2023_06_06"
        threshold_file = None

    elif batch == "kidney":
        input_file_path = "/well/rittscher/projects/PanVision/data/FullStacks/Originals/MouseKidney"
        threshold_file = None
        filenames = ["ExM42_WT_kidney_60X.tif", "ExM42_AKI_kidney_60X.tif"]

    return filenames, input_file_path, threshold_file


def filter_filenames(filenames: List[str], drugs: List[str], cells: List[int] = None) -> List[str]:
    """
    Filter filenames by requested drugs and cells
    """
    filenames = [f for drug in drugs for f in filenames if drug in f]

    # We need to filter by cell number, the problem is that 0 corresponds to no digit present in the filename
    if cells is not None:
        selected_filenames = []
        for file in filenames:
            short_file = shorten_filename(file).replace(".tif", "").replace("2nd", "")
            digits = [int(num) for num in re.findall("[0-9]+", short_file)]

            # 0 corresponds to no digit present in the filename
            if len(digits) == 0:
                if 0 in cells:
                    selected_filenames.append(file)
            elif len(digits) == 1:
                if digits[0] in cells:
                    selected_filenames.append(file)
            else:
                raise ValueError(f"Unexpected number of digits in filename {file}")
    else:
        selected_filenames = filenames

    return selected_filenames


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    success, failed = [], []
    extract_channel = 2  # 2 for normal files, 0 only works for the the kidney data!!!

    filenames, input_file_path, threshold_file = get_paths(batch=1, group=2)  # Never use group 1!
    filenames = filter_filenames(
        filenames,
        #  drugs = ['DMSO', 'Oligomycin', 'Antimycin', 'HBSS'],
        drugs=["Oligomycin"],
        #  drugs = ['DMSO'],
        #  drugs = ['Antimycin'],
        # drugs=["HBSS"],
        #  cells = [0,1,2,3,4],
        #  cells = [10, 11, 12, 13, 14],
        #  cells = [0,1,2,3,4,5,6,7,8,9],
        #  cells = [1, 4],
    )[:1]

    model_path = "/well/rittscher/users/jyo949/tmp/DrugEnsembel/AdditionalHBSSTraining/Results/"
    output_path = "/well/rittscher/users/jyo949/tmp/outputTest"

    runs = [1, 2]  # [0] means to ignore the run options

    mk_dir(output_path)
    for run in runs:
        mk_dir(os.path.join(output_path, f"Run{run}"))
        current_model_path = os.path.join(model_path, f"Run_{run}", "args.yaml") if run != 0 else model_path
        cur_model = load_model_from_config(current_model_path)
        print(f"Loaded model from {current_model_path}", flush=True)
        for file_name in filenames:
            try:
                output_file = (
                    os.path.join(output_path, f"Run{run}", shorten_filename(file_name))
                    if run != 0
                    else os.path.join(output_path, shorten_filename(file_name))
                )
                if os.path.exists(output_file):
                    print(f"File {output_file} already exists, skipping")
                    continue
                else:
                    Path(output_file).touch()
                    print(f"{'-' * 25}\n{time.strftime('%Y:%m:%d %H:%M:%S')}", shorten_filename(file_name), flush=True)

                pred = make_prediction(
                    img_path=os.path.join(input_file_path, file_name),
                    model_yaml=current_model_path,
                    model=cur_model,
                    # patch_size=[32, 1024, 1024],  # Adjust according to your GPU memory
                    # stride=[16, 512, 512],
                    patch_size=[32, 512, 512],  # Adjust according to your GPU memory
                    stride=[16, 512, 512],
                    extract_channel=extract_channel,
                    threshold_file=threshold_file,
                )

                tifffile.imwrite(
                    output_file, pred.astype("uint8"), imagej=True, metadata={"axes": "ZYX"}, compression="zlib"
                )
                print(f"{time.strftime('%Y:%m:%d %H:%M:%S')} Saved image {output_file}!", flush=True)
                success.append(file_name)
            except Exception as e:
                print(e)
                failed.append((file_name, e))

        if len(failed) != 0:
            for f in failed:
                print(f"{f[0]} failed!")
