import os
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
import torch
from torch import nn
from tqdm import tqdm

from MAPS.NoisyImmunolabeling.data import KidneyPredictDataset
from MAPS.NoisyImmunolabeling.models import FineTuneModel as Model


def add_mirrored_slices(img: torch.Tensor, n_slices: int) -> torch.Tensor:
    """
    Adds <n_slices> mirrored slices to the top and bottom of the image
    This is done to prevent artifacts at the boundaries
    """
    top_slices = img[1 : n_slices + 1]
    bottom_slices = img[-n_slices:, :, :]
    return torch.cat([top_slices.flip(0), img, bottom_slices.flip(0)], dim=0)


def remove_mirrored_slices(img: torch.Tensor, n_slices: int) -> torch.Tensor:
    """Removes <n_slices> mirrored slices from the top and bottom of the image"""
    return img[n_slices:-n_slices]


def adjust_index_for_boundary(x: int, img_shape: tuple, window_shape: tuple, dim: int) -> int:
    """
    When we get to close to a boundary in dimension <dim> so that an image of shape <window_shape>
    would be larger than the orginial <img_shape>, we move back the start coordinate enough so that
    the new crop just fits in.
    """
    if x + window_shape[dim] > img_shape[dim]:
        return max(0, img_shape[dim] - window_shape[dim])
    else:
        return x


def find_batch_boundary(start: np.array, end: np.array, window_size: np.array, stride: np.array, img_shape: tuple):
    """
    Find the start and end coordinates of the final prediction which we are filling up with the current predictions
    The idea is that we only use half of the overlap.
    However, when we are at the boundaries, we fill them up as well
    """
    overlap = window_size - stride
    shifted_start = start + overlap // 2
    shifted_end = end - overlap // 2

    # # Handle boundary cases
    start_boundary = start == 0
    shifted_start[start_boundary] = 0

    end_boundary = end == img_shape
    shifted_end[end_boundary] = img_shape[end_boundary]
    return shifted_start, shifted_end


def crop_batch(batch, start, end, window_size, stride, img_shape):
    """
    Dismiss half of the overlap and only crop out the center
    Handle boundary cases (by not cropping)
    """
    overlap = (window_size - stride) // 2
    shifted_start = overlap
    shifted_end = np.array(batch.shape)[-3:] - overlap

    # Handle boundary cases
    start_boundary = start == 0
    shifted_start[start_boundary] = 0

    end_boundary = end == img_shape
    shifted_end[end_boundary] = np.array(batch.shape)[-3:][end_boundary]

    return batch[
        ..., shifted_start[0] : shifted_end[0], shifted_start[1] : shifted_end[1], shifted_start[2] : shifted_end[2]
    ]


def predict_stack(
    img: torch.Tensor,
    model: nn.Module,
    img_window: tuple = (16, 256, 256),
    stride: tuple = (8, 128, 128),
    silent: bool = True,
) -> np.ndarray:
    """
    We use <model> to make a prediction on <img>. Because of memory constraints we have can only process
    a crop of size <img_window> at a time. In every step, this window is moved in one dimension of <stride>.
    We want <stride> to be smaller than <img_window> in every dimension as we want to have some overlap.
    This overlap is used to prevent hard cuts in the prediction. We divide the overlap in half and
    always use the half of the 'bigger side' for the final prediction.
    """
    with torch.no_grad():
        # breakpoint()
        stride = np.array(stride)
        img_window = np.array(img_window)
        final_prediction = np.zeros(img.shape, dtype="uint8")
        img_shape = np.array(img.squeeze().shape)

        for z in tqdm(range(0, img_shape[-3] - stride[0] + stride[0], stride[0]), disable=silent):
            z = adjust_index_for_boundary(z, img_shape, img_window, 0)
            for x in tqdm(range(0, img_shape[-2] - img_window[1] + stride[1], stride[1]), leave=False, disable=silent):
                x = adjust_index_for_boundary(x, img_shape, img_window, 1)
                for y in range(0, img_shape[-1] - img_window[2] + stride[2], stride[2]):
                    y = adjust_index_for_boundary(y, img_shape, img_window, 2)
                    # Get positions
                    top_left = np.array([z, x, y])
                    bottom_right = top_left + img_window

                    # Extract input batch and make predictions
                    img_batch = img[z : z + img_window[0], x : x + img_window[1], y : y + img_window[2]].to(
                        model.device
                    )
                    model.only_inner_structure = True
                    pred_batch = model.predict(img_batch[None, None])[0, 0]
                    # breakpoint()
                    pred_batch = crop_batch(pred_batch, top_left, bottom_right, img_window, stride, img_shape)

                    start, end = find_batch_boundary(
                        start=top_left, end=bottom_right, window_size=img_window, stride=stride, img_shape=img_shape
                    )

                    # Insert the current batch in the final prediction
                    final_prediction[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = (
                        pred_batch.cpu().numpy().astype("uint8")
                    )

    return final_prediction


def raw_predict_stack(
    img: torch.Tensor, model: nn.Module, n_classes: int, img_window: tuple, stride: tuple
) -> torch.Tensor:
    """
    Saves the predictions for all classes in separate channels.
    We use <model> to make a prediction on <img>. Because of memory constraints we have can only process
    a crop of size <img_window> at a time. In every step, this window is moved in one dimension of <stride>.
    We want <stride> to be smaller than <img_window> in every dimension as we want to have some overlap.
    This overlap is used to prevent hard cuts in the prediction. We divide the overlap in half and
    always use the half of the 'bigger side' for the final prediction.
    """
    with torch.no_grad():
        stride = np.array(stride)
        img_window = np.array(img_window)
        final_prediction = np.zeros((n_classes, *img.shape))
        img_shape = np.array(img.shape)

        for z in tqdm(range(0, img_shape[-3] - stride[0] + 1, stride[0])):
            z = adjust_index_for_boundary(z, img_shape, img_window, 0)
            for x in tqdm(range(0, img_shape[-2] - img_window[1] + 1, stride[1]), leave=False):
                x = adjust_index_for_boundary(x, img_shape, img_window, 1)
                for y in range(0, img_shape[-1] - img_window[2] + 1, stride[2]):
                    y = adjust_index_for_boundary(y, img_shape, img_window, 2)
                    top_left = np.array([z, x, y])
                    bottom_right = top_left + img_window

                    img_batch = img[z : z + img_window[0], x : x + img_window[1], y : y + img_window[2]].to(
                        model.device
                    )
                    if n_classes > 1:
                        pred_batch = torch.softmax(model.forward(img_batch[None, None])[0], dim=0)
                    else:
                        pred_batch = torch.sigmoid(model.forward(img_batch[None, None])[0])
                    pred_batch = crop_batch(pred_batch, top_left, bottom_right, img_window, stride, img_shape)

                    start, end = find_batch_boundary(
                        start=top_left, end=bottom_right, window_size=img_window, stride=stride, img_shape=img_shape
                    )

                    # Insert the current batch in the final prediction
                    final_prediction[..., start[0] : end[0], start[1] : end[1], start[2] : end[2]] = (
                        pred_batch.cpu().numpy()
                    )

    return final_prediction


class FakeModel:
    """
    For debugging
    """

    def predict(self, x):
        return torch.ones(x.shape).cpu().numpy().astype("uint8") * 255


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def shorten_filename(filename: str) -> str:
    return (
        filename.replace("_TOM20647_Mitotracker_NHSester488", "")
        .replace(".ims Resolution Level 1", "")
        .replace(".ims_Resolution_Level_1", "")
    )


def load_model(model_path: str, device: int) -> torch.nn.Module:
    """
    Builds a model from a configuration file and loads the best model.
    """

    # Build and load best model
    model = Model.load_from_checkpoint(model_path)
    model = model.to(torch.device(f"cuda:{device}"))
    model.eval()

    return model


def make_prediction(
    img_path: str,
    model: torch.nn.Module,
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

    prediction_window_size = patch_size
    prediction_stride = stride
    print(f"Using patch size: {prediction_window_size} with stride {prediction_stride}", flush=True)

    # Make prediction
    dataset = KidneyPredictDataset(
        imgs_path_list=[img_path],
        extract_channel=extract_channel,
        mode="test",
    )
    data = dataset.img_list[0]
    data = add_mirrored_slices(data, prediction_window_size[0] // 2)
    prediction = predict_stack(data, model, img_window=prediction_window_size, stride=prediction_stride)
    prediction = remove_mirrored_slices(prediction, prediction_window_size[0] // 2)

    return prediction


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_path", type=str)
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--input_file_path", type=str, default="")
    argparser.add_argument("--input_files_json", type=str, default="")
    argparser.add_argument("--cube_width", type=int, default=512)
    argparser.add_argument("--cube_depth", type=int, default=32)
    argparser.add_argument("--stride_width", type=int, default=256)
    argparser.add_argument("--stride_depth", type=int, default=16)
    argparser.add_argument("--nhs_channel", type=int, default=1)
    argparser.add_argument("--device", type=int, default=0)
    args = argparser.parse_args()

    if os.path.isfile(args.input_file_path):
        filenames = [(os.path.basename(args.input_file_path), os.path.dirname(args.input_file_path))]
    if args.input_files_json:
        with open(args.input_files_json, "r") as f:
            input_files = json.load(f)
            filenames = [(e["file_name"], e["path_base"]) for e in input_files]

    mk_dir(args.output_path)
    cur_model = load_model(args.model_path, args.device)
    print(f"Loaded model from {args.model_path}", flush=True)
    for file_name, input_file_path in filenames:
        output_file = os.path.join(args.output_path, shorten_filename(file_name))

        if os.path.exists(output_file):
            print(f"File {output_file} already exists, skipping")
            continue
        else:
            Path(output_file).touch()
            print(f"{'-' * 25}\n{time.strftime('%Y:%m:%d %H:%M:%S')}", shorten_filename(file_name), flush=True)
        pred = make_prediction(
            img_path=os.path.join(input_file_path, file_name),
            model=cur_model,
            patch_size=[args.cube_depth, args.cube_width, args.cube_width],  # Check with your memory
            stride=[args.stride_depth, args.stride_width, args.stride_width],
            extract_channel=args.nhs_channel,
            threshold_file=None,
        )
        tifffile.imwrite(output_file, pred.astype("uint8"), imagej=True, metadata={"axes": "ZYX"}, compression="zlib")

        print(f"{time.strftime('%Y:%m:%d %H:%M:%S')} Saved image {output_file}!", flush=True)
