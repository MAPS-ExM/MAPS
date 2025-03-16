import json
import os
import re
import time
from dataclasses import dataclass
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


def load_model_from_config(model_path: str, model_prefix: str, run: str) -> torch.nn.Module:
    """
    Builds a model from a configuration file and loads the best model.
    """
    model_yaml = os.path.join(model_path, f"{model_prefix}_{run}", "args.yaml") if run != 0 else model_path
    
    args = Args(base_yaml_file=model_yaml, extension_yaml_file=None)
    args.log_results = False  # Make sure we don't overwrite anything
    args.path_output = model_path
    args.run_name = f"{model_prefix}_{run}" 
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


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def make_ensemble_predictions(files, 
                     nhs_channel, 
                     output_path, 
                     model_path, 
                     model_prefix, 
                     model_suffix,
                     patch_depth = 32, patch_width = 1024,
                     stride_depth = 16, stride_width = 512):

    success, failed = [], []
    mk_dir(output_path)
    for run in model_suffix:
        mk_dir(os.path.join(output_path, f"{model_prefix}_{run}"))
        cur_model = load_model_from_config(model_path, model_prefix, run)
        print(f"Loaded model from {model_prefix}_{run}", flush=True)
        for cur_file in files:
            file_name = cur_file.file_name
            try:
                output_file = (
                    os.path.join(output_path, f"{model_prefix}_{run}", shorten_filename(file_name))
                    if run != 0
                    else os.path.join(output_path, shorten_filename(file_name))
                )
                if os.path.exists(output_file):
                    print(f"File {output_file} already exists, skipping")
                    continue
                else:
                    Path(output_file).touch()
                    print(f"{'-' * 25}\n{time.strftime('%Y:%m:%d %H:%M:%S')}", shorten_filename(file_name), flush=True)

                model_yaml = os.path.join(model_path, f"{model_prefix}_{run}", "args.yaml") if run != 0 else model_path
                pred = make_prediction(
                    img_path=os.path.join(cur_file.input_file_path, file_name),
                    model_yaml=model_yaml,
                    model=cur_model,
                    patch_size=[patch_depth, patch_width, patch_width],  # Adjust according to your GPU memory
                    stride=[stride_depth, stride_width, stride_width],
                    extract_channel=nhs_channel,
                    threshold_file=None,
                )

                tifffile.imwrite(
                    output_file, pred.astype("uint8"), imagej=True, metadata={"axes": "ZYX"}, compression="zlib"
                )
                print(f"{time.strftime('%Y:%m:%d %H:%M:%S')} Saved image {output_file}!", flush=True)
                success.append(file_name)
            except Exception as e:
                print(e)
                failed.append((file_name, e))

        if failed:
            for f in failed:
                print(f"{f[0]} failed!")

@dataclass
class File:
    file_name: str
    input_file_path: str
    threshold_file: Optional[str] = None


if __name__ == "__main__":
    import argparse
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--files",
        type=str,
        help="""JSON file specifying the training dataset. Example in data/Data3D.py looking like
        [
            {
                "file_name": "WT1_D6_60X.tif",
                "input_file_path": "/<ourPath>/MouseKidney_April2024/",
            }
    ]""",
    )
    argparser.add_argument("--nhs_channel", type=int, help="Which channel contains the NHS pan-staining.", default=2)
    argparser.add_argument("--model_path", type=str, help="Path where the model ensemble lives")
    argparser.add_argument("--model_prefix", type=str, help="Prefix  of the dirctory in which the models live", default='Run')
    argparser.add_argument("--model_suffix", type=int, nargs='+', help="Suffix of the dirctory in which the models live", default=[1,2,3,4,5,6])
    argparser.add_argument("--output_path", type=str, help="Path to where to save the predictiosn")
    argparser.add_argument("--patch_depth", type=int, help="Size of prediction patch", default=32)
    argparser.add_argument("--patch_width", type=int, help="Size of prediction patch", default=1024)
    argparser.add_argument("--stride_depth", type=int, help="Size of prediction stride", default=16)
    argparser.add_argument("--stride_width", type=int, help="Size of prediction stride", default=512)
    args = argparser.parse_args()

    with open(args.files, 'r') as file_dict:
        files = json.load(file_dict)
        files = [File(**f) for f in files]

    make_ensemble_predictions(
        files=files,
        nhs_channel=args.nhs_channel,
        output_path=args.output_path,
        model_path=args.model_path,
        model_prefix=args.model_prefix,
        model_suffix=args.model_suffix,
        patch_depth=args.patch_depth,
        patch_width=args.patch_width,
        stride_depth=args.stride_depth,
        stride_width=args.stride_width,
    )
