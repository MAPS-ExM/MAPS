import torch
from torch import nn
import numpy as np
import argparse
from data.BasicDataset import BasicDataset

from MAPS.MitoTracker.utils import Args

import tifffile
from tqdm import tqdm
from typing import Optional

try:
    # from models.build_model import build_model
    from MAPS.MitoTracker.models import build_model
except ModuleNotFoundError:
    pass


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
    drug_treatment: Optional[torch.Tensor] = None,
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
                    if drug_treatment is not None:
                        drug_treatment = drug_treatment.to(model.device)
                        pred_batch = model.predict(img_batch[None, None], drug_treatment[None])[0, 0]
                    else:
                        pred_batch = model.predict(img_batch[None, None])[0, 0]
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
    def predict(self, x):
        return torch.ones(x.shape).cpu().numpy().astype("uint8") * 255


if __name__ == "__main__":

    def get_args():
        parser = argparse.ArgumentParser(
            description="Predict masks from input images", formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--model_path", "-m", metavar="FILE", help="Specify the file in which the model is stored")
        # parser.add_argument('--model_name', '-mn', metavar='INPUT', nargs='+',
        #                     help='Model type used', required=True)   # Unnecessary as it is in args
        parser.add_argument(
            "--model_spec",
            "-ms",
            metavar="INPUT",
            nargs="+",
            help="Path to Model specifications yaml file",
            required=True,
        )
        parser.add_argument(
            "--input_path", "-i", metavar="INPUT", nargs="+", help="filenames of input images", required=True
        )
        parser.add_argument("--output_path", "-o", metavar="INPUT", nargs="+", help="Filenames of ouput images")
        parser.add_argument(
            "--mask_threshold",
            "-t",
            type=float,
            help="Minimum probability value to consider a mask pixel white",
            default=0.5,
        )
        parser.add_argument("--scale", "-s", type=float, help="Scale factor for the input images", default=1)
        parser.add_argument(
            "-pn", "--pix_num", dest="pix_num", type=int, default=1000, help="Minimum number of voxels in an instance"
        )
        parser.add_argument(
            "--batch_size", type=int, default=[32, 128, 128], nargs="+", help="size of the sliding window"
        )
        parser.add_argument(
            "stride", type=int, default=[16, 64, 64], nargs="+", help="stride by which we move the sliding window"
        )
        parser.add_argument(
            "-sp",
            "--start_point",
            dest="start_point",
            type=int,
            default=(0, 0, 0),
            nargs="+",
            help="start coordinates for sliding window",
        )
        parser.add_argument(
            "-ep",
            "--end_point",
            dest="end_point",
            type=int,
            default=(200, 2048, 2048),
            nargs="+",
            help="end coordinates for sliding window ",
        )
        parser.add_argument("--device", default="cpu")
        return parser.parse_args()

    class DebugArgs:
        def __init__(self):
            self.model_path = "/well/rittscher/users/jyo949/PanVisionSeg/output/DebugOneBatch/BestModel.pt"
            self.model_name = "BasicUNet3D"
            self.model_spec = "/well/rittscher/users/jyo949/PanVisionSeg/output/DebugOneBatch/args.yaml"
            self.input_path = "/well/rittscher/projects/PanVision/data/training_data_Sept2022/dragonfly/Hela_Tom20_MitoOrange_2022-07-02_2.ims_Resolution_Level_1_nhs_norm.tif"
            self.output_path = "pred_test.tif"
            self.batch_size = [8, 64, 64]
            self.stride = [4, 32, 32]
            self.mask_threshold = 0.5
            self.device = "cuda:0"

    # Load arguments
    script_args = get_args()  # DebugArgs()
    args = Args(script_args.model_spec)
    args.__dict__.update(script_args.__dict__)  # Add and overwrite args with the script_args

    # Load data
    data = BasicDataset(imgs_dir=[args.input_path], scale=args.scale)
    data = data.img_list[0].to(args.device)
    print("Data shape: {data.shape}")

    # Load model
    model = build_model(args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device)["model_state_dict"])
    model.to(args.device)

    # For Debugging
    # data = torch.zeros(32, 256, 256)
    # model = FakeModel()

    # Make prediction
    print("Make predicion")
    pred = predict_stack(data, model, args.batch_size, args.stride)

    # Save prediction
    tifffile.imwrite(args.output_path, pred)
    print("Image saved")
