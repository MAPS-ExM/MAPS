"""3D Unet"""

import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from MAPS.utils import SegmentatorNetwork, UNet, UNetGN, UNeXt

from MAPS.MitoTracker.utils.Losses import DiceLoss, GeneralisedCE
from typing import Dict


class BasicUNet3D(nn.Module):
    def __init__(
        self,
        n_classes: int,
        loss_fn: _Loss = None,
        input_channels: int = 1,
        mask_threshold: float = 0.5,
        device: torch.cuda.device = "cpu",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.embedding_network = None
        self.segmentation_head = SegmentatorNetwork(n_classes)
        self.device = device

        # Build loss function
        self.selected_loss_fn = loss_fn
        if self.selected_loss_fn["Dice"]:
            self.dice = DiceLoss(mode="binary" if n_classes == 1 else "multiclass", ignore_index=-1)
        if self.selected_loss_fn["CE"]:
            self.CE = nn.CrossEntropyLoss(ignore_index=-1)
        if self.selected_loss_fn["BCE"]:
            self.bce = nn.BCEWithLogitsLoss()
        if self.selected_loss_fn["GeneralisedCE"]:
            self.generalCE = GeneralisedCE(q=0.8, indexToIgnore=None)
        if self.selected_loss_fn["MSE"]:
            self.mse = nn.MSELoss()

        # For prediction
        self.mask_threshold = mask_threshold

    @staticmethod
    def unpack_batch(batch, item, device=None):
        """
        Gets called in the evaluation to extract the raw image from the batch to make a prediction
        """
        res = batch[item]
        if device is not None:
            res = res.to(device)
        return res

    def forward(self, x):
        return self.segmentation_head(self.embedding_network(x))

    def comp_loss(self, batch, iteration=0, tracker=None, mode="train", model=None):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        img, target = batch["image"], batch["target"]
        # features = self.embedding_network(img)
        # pred = self.segmentation_head(features)
        if model is not None:
            pred = model.__call__(img)  # Need this for the DataParallel to work
        else:
            pred = self.__call__(img)

        loss = self._comp_loss(pred, target)
        tracker.add_scalar(f"Loss ({mode})", loss.detach().cpu().item(), iteration, mode)
        return loss

    def _comp_loss(self, pred, target):
        loss = 0
        # Loss selection
        if self.selected_loss_fn["CE"]:
            loss += self.CE(pred, target.squeeze(1))
        if self.selected_loss_fn["BCE"]:
            loss += self.bce(pred, target)
        if self.selected_loss_fn["GeneralisedCE"]:
            loss += self.generalCE(pred, target)
        if self.selected_loss_fn["Dice"]:
            loss += self.dice(pred, target)
        if self.selected_loss_fn["MSE"]:
            loss += self.mse(torch.sigmoid(pred), target / 255)
        return loss

    def predict_batch(self, batch: Dict[str, torch.Tensor]):
        x = self.unpack_batch(batch, "image", self.device)
        # return self._predict(x)
        return self.predict(x)

    def predict(self, x: torch.Tensor):
        if self.n_classes == 1:
            return torch.sigmoid(self.forward(x)) > self.mask_threshold
        else:
            return torch.argmax(self.forward(x), dim=1, keepdim=True)


class BasicSmallUNet(BasicUNet3D):
    def __init__(
        self,
        n_classes: int,
        loss_fn: _Loss = None,
        mask_threshold: float = 0.5,
        loss_weight=None,
        pretrain: str = "",
        input_channels: int = 1,
        residual: bool = False,
        device: torch.cuda.device = "cpu",
    ):
        super().__init__(n_classes, loss_fn, mask_threshold=mask_threshold, device=device)
        self.embedding_network = UNet(
            encoder_channels=[input_channels, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 1],
            residual=residual,
            type="3D",
        )
        self.segmentation_head = SegmentatorNetwork(n_classes, in_classes=32)


class GNSmallUNet(BasicUNet3D):
    def __init__(
        self,
        n_classes: int,
        loss_fn: _Loss = None,
        mask_threshold: float = 0.5,
        residual: bool = False,
        device: torch.cuda.device = "cpu",
    ):
        super().__init__(n_classes, loss_fn, mask_threshold=mask_threshold, device=device)
        self.embedding_network = UNetGN(
            encoder_channels=[1, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 1],
            residual=residual,
            type="3D",
        )
        self.segmentation_head = SegmentatorNetwork(n_classes, in_classes=32)


class SmallUNeXt(BasicUNet3D):
    def __init__(
        self,
        n_classes: int,
        loss_fn: _Loss = None,
        mask_threshold: float = 0.5,
        residual: bool = False,
        device: torch.cuda.device = "cpu",
    ):
        super().__init__(n_classes, loss_fn, mask_threshold=mask_threshold, device=device)
        self.embedding_network = UNeXt(
            encoder_channels=[1, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 1],
            residual=residual,
            type="3D",
        )
        self.segmentation_head = SegmentatorNetwork(n_classes, in_classes=32)


class BasicSmallRegressionUNet(BasicUNet3D):
    def __init__(
        self, n_classes: int, loss_fn: _Loss = None, residual: bool = False, device: torch.cuda.device = "cpu"
    ):
        super().__init__(n_classes, loss_fn, mask_threshold=0, device=device)
        self.embedding_network = UNet(
            encoder_channels=[1, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 1],
            residual=residual,
            type="3D",
        )
        self.segmentation_head = SegmentatorNetwork(n_classes, in_classes=32)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class ConditionalGNUNet(BasicUNet3D):
    def __init__(
        self,
        n_classes: int,
        loss_fn: _Loss = None,
        mask_threshold: float = 0.5,
        residual: bool = False,
        cat_emb_dim: int = 8,
        device: torch.cuda.device = "cpu",
    ):
        super().__init__(n_classes, loss_fn, mask_threshold=mask_threshold, device=device)
        self.embedding_network = UNetGN(
            encoder_channels=[1, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 1],
            residual=residual,
            type="3D",
            cat_emb_dim=cat_emb_dim,
        )

        self.segmentation_head = SegmentatorNetwork(n_classes, in_classes=32)

        # For the embedding I want the classes: None, HBSS, Antimycin, Oligomycin
        # Later to test: sample with a certain probability the 'All' class to make sure that the model still can generalise
        self.cat_emb_dim = cat_emb_dim
        self.drug_embedding = nn.Embedding(5, cat_emb_dim)
        self.drug_None_prob = 0.5

    def comp_loss(self, batch, iteration=0, tracker=None, mode="train", model=None):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        img, target, drug_treatment = batch["image"], batch["target"], batch["drug_treatment"]
        drug_embedding = self.drug_embedding(drug_treatment)

        tracker.add_scalar(
            f"Prob_DrugNone",
            np.clip(0.9 - iteration * (0.9 - self.drug_None_prob) / 2500, self.drug_None_prob, 0.9),
            iteration,
            mode,
        )
        if random.random() < np.clip(
            0.9 - iteration * (0.9 - self.drug_None_prob) / 2500, self.drug_None_prob, 0.9
        ):  # We go from 10% to 90% drug information during the first 2.5k iterations
            drug_treatment = None

        features = self.embedding_network(img, cat_embedding=drug_embedding)
        pred = self.segmentation_head(features)

        loss = self._comp_loss(pred, target)
        tracker.add_scalar(f"Loss ({mode})", loss.detach().cpu().item(), iteration, mode)
        return loss

    def predict_batch(self, batch: Dict[str, torch.Tensor]):
        x = self.unpack_batch(batch, "image", self.device)
        drug_treament = self.unpack_batch(batch, "drug_treatment", self.device)
        return self.predict(x, drug_treament)

    def predict(self, x, drug_treatment=None):
        drug_embedding = self.drug_embedding(drug_treatment) if drug_treatment is not None else None
        features = self.embedding_network(x, cat_embedding=drug_embedding)
        pred = self.segmentation_head(features)
        if self.n_classes == 1:
            return torch.sigmoid(pred) > self.mask_threshold
        else:
            return torch.argmax(pred, dim=1, keepdim=True)
