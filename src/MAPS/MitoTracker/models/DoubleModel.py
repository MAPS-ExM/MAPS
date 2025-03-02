"""3D Unet"""

import sys

sys.path.insert(0, "../..")

import random
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.modules.loss import _Loss
from utils.common_utils import GPU_Dilator
from utils.Losses import DiceLoss

from MAPS.utils import SegmentatorNetwork, UNet, UNetGN


def _disable_require_grad(model):
    for param in model.parameters():
        param.requires_grad = False


class DoubleModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        loss_fn: _Loss = None,
        mask_threshold: float = 0.5,
        input_channels: int = 1,
        loss_weight=None,
        pretrain: str = "",
        residual: bool = True,
        cat_emb_dim: Optional[int] = None,
        device: torch.cuda.device = "cpu",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.conditional = cat_emb_dim is not None  # Should the model make use of  the class information?
        self.cat_emb_dim = cat_emb_dim
        self.drug_None_prob = 0.5
        self.only_inner_structure = False
        self.return_stage1_pred = False

        if pretrain:
            import os

            from utils.Args import Args

            from .build_model import build_model

            pre_trained_args = Args(os.path.join(pretrain, "args.yaml"))
            pre_trained_args.device = device
            pretrained = build_model(pre_trained_args)
            pretrained_dict = os.path.join(pre_trained_args.path_output, pre_trained_args.run_name, "BestModel.pt")
            # pretrained.load_state_dict(torch.load(pretrained_dict, map_location=device)['model_state_dict'])
            state_dict = torch.load(pretrained_dict, map_location=device)["model_state_dict"]
            try:  # Some legacy code to get a pretrained conditional model running
                state_dict["embedding_network.encoder.layers.Down0.mlp_cat.weight"] = state_dict[
                    "embedding_network.encoder.layers.Down0.mlp_cat.0.weight"
                ]
                state_dict["embedding_network.encoder.layers.Down0.mlp_cat.bias"] = state_dict[
                    "embedding_network.encoder.layers.Down0.mlp_cat.0.bias"
                ]
                # Delete keys
                del state_dict["embedding_network.encoder.layers.Down0.mlp_cat.0.weight"]
                del state_dict["embedding_network.encoder.layers.Down0.mlp_cat.0.bias"]
            except KeyError:
                pass
            pretrained.load_state_dict(state_dict)

            # Freeze the labels
            _disable_require_grad(pretrained.embedding_network)
            try:
                _disable_require_grad(pretrained.drug_embedding)
            except AttributeError:
                pass
            _disable_require_grad(pretrained.segmentation_head)

            self.pre_trained = pretrained
            print(f"Loaded pretrained model from {pretrain}")
        else:
            print("No pretrained model given but required for DoubleModel")

        if (
            type(self.pre_trained).__name__ == "BasicSmallUNet"
        ):  # SmallAltUNet for legacy code in order to be able to use old models for prediction
            self.fine_decoder = UNet(
                encoder_channels=[input_channels, 64, 128, 256, 512],
                decoder_channels=[512, 256, 128, 64, 32, 1],
                type="3D",
            ).decoder
        elif type(self.pre_trained).__name__ == "GNSmallUNet":
            self.fine_decoder = UNetGN(
                encoder_channels=[1, 64, 128, 256, 512],
                decoder_channels=[512, 256, 128, 64, 32, 1],
                residual=residual,
                type="3D",
            ).decoder
        elif type(self.pre_trained).__name__ == "ConditionalGNUNet":
            self.fine_decoder = UNetGN(
                encoder_channels=[1, 64, 128, 256, 512],
                decoder_channels=[512, 256, 128, 64, 32, 1],
                residual=residual,
                type="3D",
                cat_emb_dim=cat_emb_dim,
            ).decoder
            self.drug_embedding = torch.nn.Embedding(5, self.cat_emb_dim)
        else:
            self.fine_decoder = UNet(
                encoder_channels=[1, 64, 128, 256, 512],
                decoder_channels=[512, 256, 128, 64, 32, 1],
                residual=residual,
                type="3D",
            ).decoder

        self.fine_seg_head = SegmentatorNetwork(n_classes, in_classes=32)
        self.dilator = GPU_Dilator(25).to(device)

        self.device = device

        # Build loss function
        self.indexToIgnore = -1
        self.selected_loss_fn = loss_fn
        if self.selected_loss_fn["Dice"]:
            self.dice = DiceLoss(
                mode="binary" if n_classes == 1 else "multiclass",
                ignore_index=self.indexToIgnore,
            )
        if self.selected_loss_fn["CE"] or self.selected_loss_fn["CE_PunishAmbig001"]:
            self.CE = nn.CrossEntropyLoss(ignore_index=self.indexToIgnore)
        if self.selected_loss_fn["BCE"]:
            self.bce = nn.BCEWithLogitsLoss()

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

    def expand_pretrain_mitotracker(self, pred: torch.Tensor):
        assert pred.shape[1] == 1, "Expected only one channel in the prediction"
        for b in range(pred.shape[0]):
            # Extract the inner prediction
            mitoTracker_pred = (pred[b, 0] == 2).float()
            mitoTracker_pred = self.dilator(mitoTracker_pred)

            # Add the expanded inner prediction to the outer prediction
            outer_mito = torch.logical_or(pred[b, 0] > 0, mitoTracker_pred > 0).float()
            outer_mito = outer_mito + (pred[b, 0] == 2).float()
            pred[b] = outer_mito[None]
        return pred

    def forward(self, x, drug_treatment=None):
        with torch.no_grad():
            self.pre_trained.eval()
            drug_embedding_pretrain = (
                self.pre_trained.drug_embedding(drug_treatment) if drug_treatment is not None else None
            )
            latent_features = [x] + self.pre_trained.embedding_network.encoder(x, cat_embedding=drug_embedding_pretrain)
            if not self.only_inner_structure:
                final_features = self.pre_trained.embedding_network.decoder(
                    latent_features, cat_embedding=drug_embedding_pretrain
                )
                pretrained_pred = self.pre_trained.segmentation_head(final_features)
                if self.return_stage1_pred:
                    return pretrained_pred
                pretrained_pred = torch.argmax(pretrained_pred, dim=1, keepdim=True)
                pretrained_pred = self.expand_pretrain_mitotracker(pretrained_pred)

        drug_embedding = self.drug_embedding(drug_treatment) if drug_treatment is not None else None
        features = self.fine_decoder(latent_features, cat_embedding=drug_embedding)
        pred = self.fine_seg_head(features)
        # Set the mask by the pre-trained model to background
        if not self.only_inner_structure:
            pred[pretrained_pred.expand(pred.shape) == 0] = 0  # All other channels
            pred[:, 0][pretrained_pred.squeeze(1) == 0] = 1  # Background
        return pred

    def comp_loss(self, batch, iteration=0, tracker=None, mode="train", model=None):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        loss = 0
        img, target = batch["image"], batch["target"]
        drug_treatment = batch["drug_treatment"] if self.conditional else None

        # I want the effect of the drug information to be a residual effect
        # Therefore I start the training with little drug information and increase it over time

        # Get random bool
        tracker.add_scalar(
            f"Prob_DrugNone",
            np.clip(
                0.9 - iteration * (0.9 - self.drug_None_prob) / 2500,
                self.drug_None_prob,
                0.9,
            ),
            iteration,
            mode,
        )
        if random.random() < np.clip(
            0.9 - iteration * (0.9 - self.drug_None_prob) / 2500,
            self.drug_None_prob,
            0.9,
        ):  # We go from 10% to 90% drug information during the first 2.5k iterations
            drug_treatment = None

        # Get predictions from pretrained model
        with torch.no_grad():
            self.pre_trained.eval()
            drug_embedding_pretrained = (
                self.pre_trained.drug_embedding(drug_treatment)
                if (self.conditional and drug_treatment is not None)
                else None
            )
            latent_features = [img] + self.pre_trained.embedding_network.encoder(
                img, cat_embedding=drug_embedding_pretrained
            )
            final_features = self.pre_trained.embedding_network.decoder(
                latent_features, cat_embedding=drug_embedding_pretrained
            )
            pretrained_pred = self.pre_trained.segmentation_head(final_features)
            pretrained_pred = torch.argmax(pretrained_pred, dim=1, keepdim=True)
            pretrained_pred = self.expand_pretrain_mitotracker(pretrained_pred)

        drug_embedding = self.drug_embedding(drug_treatment) if drug_treatment is not None else None
        features = self.fine_decoder(latent_features, cat_embedding=drug_embedding)
        pred = self.fine_seg_head(features)

        # Only consider the elements where pretrained model predicted something
        target[pretrained_pred.expand(target.shape) == 0] = self.indexToIgnore

        # Loss selection
        if self.selected_loss_fn["CE"]:
            loss += self.CE(pred, target.squeeze(1))
        if self.selected_loss_fn["CE_PunishAmbig001"]:
            loss += self.CE(pred, target.squeeze(1))
            # Predicted ambiguous but truely M or C
            pred_flat = rearrange(pred, "b c h w d -> (b h w d) c")
            target_flat = target.flatten()
            pred_ambiguous = torch.argmax(pred_flat, dim=1) == 4
            true_MorC = torch.logical_or(target_flat == 2, target_flat == 3)
            indices = torch.logical_and(pred_ambiguous, true_MorC)
            loss += (
                0.01
                * (indices.sum() / len(indices))
                * F.cross_entropy(pred_flat[indices, ...], target_flat[indices], reduction="sum")
            )
        if self.selected_loss_fn["BCE"]:
            loss += self.bce(pred, target)
        if self.selected_loss_fn["Dice"]:
            loss += self.dice(pred, target)

        tracker.add_scalar(f"Loss ({mode})", loss.detach().cpu().item(), iteration, mode)

        return loss

    def predict(self, x, drug_treatment=None):
        if self.n_classes == 1:
            raise NotImplementedError()
        else:
            logits = self.forward(x, drug_treatment=drug_treatment)
            if self.only_inner_structure:
                logits[:, 0] = -np.inf  # Ignore background
                logits[:, 1] = -np.inf  # Ignore intermembrane space
            pred = torch.argmax(logits, dim=1, keepdim=True)

            if self.return_stage1_pred:
                pred = self.expand_pretrain_mitotracker(pred)
        return pred

    def predict_batch(self, batch: Dict[str, torch.Tensor]):
        x = self.unpack_batch(batch, "image", self.device)
        drug_treament = self.unpack_batch(batch, "drug_treatment", self.device) if self.conditional else None
        return self.predict(x, drug_treament)

    # def predict(self, x, drug_treatment = None):
    #     pred = self.forward(x, drug_treatment=drug_treatment)
    #     # drug_embedding = self.drug_embedding(drug_treatment) if drug_treatment is not None else None
    #     # features = self.embedding_network(x, cat_embedding=drug_embedding)
    #     # pred = self.segmentation_head(features)
    #     if self.n_classes == 1:
    #         return torch.sigmoid(pred) > self.mask_threshold
    #     else:
    #         return torch.argmax(pred, dim=1, keepdim=True)
