import copy
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# from .Metrics import Metrics
from MAPS.MitoTracker.utils import (
    Args,
    Metrics,
)

from MAPS.MitoTracker.utils.predict import (
    add_mirrored_slices,
    predict_stack,
    raw_predict_stack,
    remove_mirrored_slices,
)

from .common_utils import save_state


class FakeSummaryWriter:
    """
    A SummaryWriter like class that just does nothing. Used as a dummy if I don't want to log any results
    """

    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


def init_writer(log_results, log_dir):
    """Wrapper to switch between real and fake summary writer (that just does nothing)"""
    if log_results:
        return SummaryWriter(log_dir)
    else:
        return FakeSummaryWriter


class Tracker:
    """
    Class to keep track of all relevant hyperparameters and metrics
    To read all descriptions specified in the args.yaml files, navigate to the output directory and type
    find . -name 'description.txt' | xargs grep '^desc.*'
    """

    def __init__(
        self,
        path_output: str,
        log_results: bool,
        base_config: Optional[str] = None,
        extension_config: Optional[str] = None,
        args: Optional[Args] = None,
    ):  # type: ignore
        self.path_output = path_output

        # Record start time
        start_time = datetime.now().strftime("%m_%d_%H_%M")
        self.start_time = start_time
        args.start_time = start_time
        self.run_name = args.run_name
        self.set_up_dirs(log_results, path_output, base_config, extension_config, args)
        self.writer = init_writer(log_results, log_dir=os.path.join(path_output, args.run_name))

        # Model evaluation
        self.eval_iter = 0
        self.binary_classification = args.num_classes == 1
        self.num_classes = args.num_classes
        self.n_input_channels = getattr(args, "input_channels", 1)

        # Highest values
        self.best_loss = 1e9
        self.max_iou = 0
        self.max_dice = 0
        self.max_acc = 0
        self.path_output = args.path_output
        self.run_name = args.run_name

        # Buffer for writing to summary writer
        self.buffer = {}

        # Early stopping
        self.eval_iterLastImprovement = 0

        # Test files
        self.test_files = getattr(args, "path_img_test", [])

    def set_up_dirs(self, log_results: bool, path_output: str, base_config: str, extension_config: str, args: Args):  # type: ignore
        """
        Sets up an output directory with the name of the run. If the directory already exists, the old one is
        copied into a subdirectory "old_experiements" and a new one is created.
        Saves a description of the arguments and saves the args.yamls and extension file for easy reproducibility
        """
        # Set up directories
        if log_results:
            try:
                os.makedirs(os.path.join(path_output, args.run_name))
            except AttributeError as err:
                logging.info(f"No name for the run given in the args file! \n {err}")
                sys.exit()
            except FileExistsError:
                logging.info(
                    f"There is already a directory with the same name!{os.path.join(path_output, args.run_name)}"
                )
                if not args.run_name.lower().startswith("debug"):
                    self._move_results_dir(path_output, args.run_name)  # Move old directory and ..
                    os.makedirs(os.path.join(path_output, args.run_name))  # Create new directory
                    logging.info('Moved the old directory into "old_experiments"')
                    # sys.exit()

            # Save current configuration:
            with open(os.path.join(args.path_output, args.run_name, "description.txt"), "w") as f:
                description = ""
                description += args.to_string()
                f.write(description)

            with open(os.path.join(args.path_output, args.run_name, "args.yaml"), "w") as f:
                yaml.dump(args.orig_args, f, default_flow_style=False)

            # Copy args file if they are present
            if os.path.isfile(base_config):
                shutil.copyfile(base_config, os.path.join(args.path_output, args.run_name, "args.yaml"))
            if extension_config is not None:
                if os.path.isfile(extension_config):
                    shutil.copyfile(
                        extension_config, os.path.join(args.path_output, args.run_name, "args_extension.yaml")
                    )

        else:
            logging.info("Results are NOT logged!")

    def _move_results_dir(self, path_dir: str, run_name: str) -> None:
        """Appends the current date to the given directory <path_dir> and moves it into 'old_experiments'"""
        path_run = os.path.join(path_dir, run_name)

        # Create 'old_experiments' directory if it doesnt exist yet
        if not os.path.exists(os.path.join(path_dir, "old_experiments")):
            os.makedirs(os.path.join(path_dir, "old_experiments"))

        # Rename current run
        creation_date = self._find_creation_date(path_run)
        new_name = path_run.rstrip("/") + "_" + creation_date + "/"
        os.rename(path_run, new_name)

        # Move the directory into 'old_experiments'
        base, run_name = os.path.split(os.path.normpath(new_name))
        shutil.move(new_name, os.path.join(base, "old_experiments", run_name))

    def _find_creation_date(self, path: str) -> str:
        """Finding the creation date apparently is tricky, so I just use the earlist modification date"""
        dates = []
        for file in os.listdir(path):
            c_path = os.path.join(path, file)
            date = datetime.fromtimestamp(os.path.getmtime(c_path)).strftime("%Y_%m_%d_%H_%M")
            dates.append(date)
        return min(dates)

    def _shorten_name(self, name: str) -> str:
        """Shortens a long run name by taking 5 character substrings between separting underscores"""
        parts = name.split("_")
        base = parts.pop(0)
        shortened_parts = []
        for part in parts:
            try:
                _ = float(part)
                shortened_parts.append(part)
            except ValueError:
                shortened_parts.append(part[-5:])
        return "_".join([base] + shortened_parts)

    def eval_model(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        save_eval_txt: bool = False,
        iteration: int = None,
        use_best_model: bool = False,
        error_tol: int = 0,
        name_prefix: str = "",
        respect_ambiguity: bool = False,
    ) -> str:
        """
        Evaluateds <model> on the data in <data_loader> and saves the model if it is better than what has been
        evaluated before.
        :param save_eval_txt: If true, saves the results in a txt file (only supposed to be used on the very last evaluation)
        :param iteration: Training iteration to be stored if this is the best model seen so far (not used elsewhere!)
        :param use_best_model: For final evaluation, possible to load the best model seen soar.
        """
        # Parameter iteration is only used to print in which training iteration the best model was found
        loss = 0
        metrics = Metrics("IoU Dice Accuracy", labels_to_eval=[1, 2, 3, 4])

        with torch.no_grad():
            if use_best_model:
                model = self.loadBestModel(model)
            model.eval()
            for batch in data_loader:
                pred_target = model.predict_batch(batch)

                # Compute loss which buffers the values to be written to the Summary Writer later
                loss += model.comp_loss(batch, iteration=self.eval_iter, tracker=self, mode="val")  # , model=model)
                target = model.unpack_batch(batch, "target", model.device)

                if "mask" in batch:
                    # Black out the masked areas by setting them to -1 as only
                    # the labels in metrics.labels_to_eval are evaluateds
                    mask = model.unpack_batch(batch, "mask", model.device)
                    pred_target[mask == 0] = False if pred_target.dtype is torch.bool else -1
                    target[mask == 0] = -1

                if self.binary_classification:
                    metrics.comp_metrics_from_bool(pred_target > 0, target > 0.5)
                else:
                    metrics.comp_metrics(pred_target, target, error_tol=error_tol, respect_ambiguity=respect_ambiguity)

        self.flush_buffer()  # Write the loss which was buffered during the compute
        loss = (loss / len(data_loader)).cpu().item()

        # Save metrics for this epoch to the summary writer
        metrics.summarise_metrics()
        cur_iou, cur_dice, cur_acc = metrics.get("iou dice accuracy")
        metrics.add_metric_tensorboard(self.writer, self.eval_iter)

        if not use_best_model:
            self.saveBestModel(model, loss, cur_iou, cur_dice, cur_acc, iteration)

        self.eval_iter += 1
        eval_results = f"Loss: {loss:.6f}, IoU: {cur_iou:.4f}, Dice: {cur_dice:.4f}, Accuracy: {cur_acc:.4f}"

        if save_eval_txt:
            if any("Experiments" in dirname for dirname in self.path_output.split("/")):
                # For Experiments I want to additionally (!) summarize all the test results in one file:
                with open(os.path.join(self.path_output, "finalTestResult.txt"), "a") as f:
                    short_name = self._shorten_name(self.run_name)
                    f.write(
                        f"{name_prefix + short_name:45}: {eval_results}, TrainingEpochs: {self.eval_iter}"
                        f"Best model from iteration {self.eval_iterLastImprovement}  -  {self.start_time}\n"
                    )
            with open(os.path.join(self.path_output, self.run_name, f"finalResult{name_prefix.strip()}.txt"), "w") as f:
                f.write(eval_results)

        return eval_results

    def compare_predictions(
        self,
        model: torch.nn.Module,
        nhs_images: List[torch.Tensor],
        annotations: List[torch.Tensor],
        img_window: list = [24, 256, 256],
        stride: list = [12, 128, 128],
    ) -> None:
        """
        Produces a .png image for every file in nhs image with three columns showing the original image,
        the annotation and the prediction. 6 randomly selected annotated slices are shown.
        Files are saved in the comparison folder of the current output directory.
        """
        path_comparison_dir = os.path.join(self.path_output, self.run_name, "comparison")
        if not os.path.exists(path_comparison_dir):
            # path_comparison_dir = os.path.join(self.path_output, self.run_name, 'comparison')
            os.makedirs(path_comparison_dir)

        model = self.loadBestModel(model)
        for ix, (img, annotation) in enumerate(zip(nhs_images, annotations)):
            # If the image is smaller than the window, add mirrored slices to the image to be able to process it in one go
            if img.shape[0] < img_window[0]:
                slices_added = True
                additional_slices = img_window[0] - img.shape[0]
                img = add_mirrored_slices(img, additional_slices)
            else:
                slices_added = False

            if self.n_input_channels == 1:
                prediction = predict_stack(img.squeeze(), model, img_window=img_window, stride=stride)
            else:
                raise NotImplementedError("We only work with one input channel!")

            if slices_added:
                prediction = remove_mirrored_slices(prediction, additional_slices)
                img = remove_mirrored_slices(img, additional_slices)

            # Find slices with annotations
            annotated_slices = torch.where(annotation.sum(dim=(1, 2)) > 0)[0]

            # Select 6 random slices with annotations
            indices = sorted(np.random.choice(annotated_slices, size=min(8, len(annotated_slices)), replace=False))

            _, ax = plt.subplots(len(indices), 3, figsize=(10, int(len(indices) * 3)))

            for i, slice in enumerate(indices):
                ax[i, 0].imshow(annotation[slice])
                ax[i, 0].set_title(f"Annotation - Slice {slice}")

                ax[i, 1].imshow(prediction[slice])
                ax[i, 1].set_title(f"Prediction - Slice {slice}")

                ax[i, 2].imshow(img[slice])
                ax[i, 2].set_title(f"NHS - Slice {slice}")
            plt.tight_layout()
            file_name = self.test_files[ix].split("/")[-1]
            plt.savefig(os.path.join(path_comparison_dir, f"{file_name[: file_name.rfind('.tif')]}.png"))
            plt.close()

    def save_model_preds(
        self,
        data: List[torch.Tensor],
        model: torch.nn.Module,
        img_window: list = [32, 128, 128],
        stride: list = [16, 64, 64],
        pred_name: List[str] = [],
        include_class_pred: bool = False,
    ) -> None:
        assert len(data) == len(pred_name), f"Need to provide a name in <pred_name> for every element in <data>"
        for img, name in zip(data, pred_name):
            self.save_model_pred(
                img,
                model=model,
                img_window=img_window,
                stride=stride,
                pred_name=name,
                include_class_pred=include_class_pred,
            )

    def save_model_pred(
        self,
        data: torch.Tensor,
        model: torch.nn.Module,
        img_window: list = [32, 128, 128],
        stride: list = [16, 64, 64],
        pred_name: str = "pred_results",
        include_class_pred: bool = False,
    ) -> None:
        model = self.loadBestModel(model)
        model.eval()
        if self.n_input_channels == 1:
            prediction = predict_stack(data, model, img_window=img_window, stride=stride)
        else:
            raise NotImplementedError("We only work with one input channel!")
        tifffile.imwrite(
            os.path.join(self.path_output, self.run_name, f"{pred_name}.tif"),
            prediction,
            imagej=True,
            metadata={"axes": "ZYX"},
            compression="zlib",
        )
        if include_class_pred:
            prediction = raw_predict_stack(data, model, self.num_classes, img_window=img_window, stride=stride)
            tifffile.imwrite(
                os.path.join(self.path_output, self.run_name, f"{pred_name}_raw.tif"),
                prediction,
                imagej=True,
                metadata={"axes": "CZYX"},
                compression="zlib",
            )

    def add_scalar(self, tag: str, value: float, niter: int, mode: str = "train"):
        """Wrapper to add a scalar to the underlying SummaryWriter"""
        # If we are in train mode but the scalar is only supposed to be added for debugging, just skip
        if mode == "debug":
            return
        self.writer.add_scalar(tag, value, niter)

    def buffer_scalar(self, tag: str, value: float, niter: int, mode: str = "train"):
        """
        I often log the results in the training loop for every iteration but only want to report the values over
        the whole epoch.
        Therefore, I'm just gonna buffer them here in a list and the next method will form the mean value and
        just pick the minimal iteration and then write them out to the SummaryWriter.
        """
        if tag not in self.buffer:
            self.buffer[tag] = {"value": [value], "niter": [niter], "mode": mode}
        else:
            self.buffer[tag]["value"].append(value)
            self.buffer[tag]["niter"].append(niter)

    def flush_buffer(self):
        """
        Computes the mean of the accumulated values in the buffer and writes them to the SummaryWriter.
        """
        for key, value_dict in self.buffer.items():
            mean_value = np.mean(value_dict["value"])
            min_iter = np.min(value_dict["niter"])  # Use the first iteration of current batch
            self.add_scalar(key, mean_value, min_iter, mode=value_dict["mode"])

        # Empty the buffer
        self.buffer = {}

    def saveBestModel(self, model, loss, cur_iou, cur_dice, cur_acc, iter):
        # # Save the model if two of these metrics are better then the old best ones
        # better = np.array([cur_iou, cur_dice, cur_acc]) > np.array([self.max_iou, self.max_dice, self.max_acc])
        # better = np.sum(better) >= 2
        if loss < self.best_loss:
            self.best_loss, self.max_iou, self.max_dice, self.max_acc = loss, cur_iou, cur_dice, cur_acc
            save_state(os.path.join(self.path_output, self.run_name, "BestModel.pt"), model)

            with open(os.path.join(self.path_output, self.run_name, "bestResult.txt"), "w") as f:
                f.write(
                    f"IoU: {self.max_iou:.4f}, Dice: {self.max_dice:.4f}, "
                    f"Accuracy: {self.max_acc:.4f} in iteration {iter}"
                )

            self.eval_iterLastImprovement = self.eval_iter

    def loadModel(self, model, path: str, device=None):
        logging.info(f"Load model from {path}")
        model = copy.deepcopy(model)
        model.load_state_dict(torch.load(path, map_location=device)["model_state_dict"], strict=False)
        model.eval()
        return model

    def loadBestModel(self, model, device=None):
        try:
            path = os.path.join(self.path_output, self.run_name, "BestModel.pt")
            model = self.loadModel(model, path, device)
            logging.info("Model with the best validation performance loaded")
        except FileNotFoundError:
            logging.info("No best model found! Using the current model")
            print(">>> No best model found! Using the current model!")
        return model

    def close(self):
        self.writer.close()
