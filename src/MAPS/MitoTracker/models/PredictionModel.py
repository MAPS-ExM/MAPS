import os
import random
import sys
from collections import namedtuple, defaultdict
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.modules.loss import _Loss

from MAPS.MitoTracker.utils.common_utils import GPU_Dilator
from MAPS.MitoTracker.utils import Args
from MAPS.MitoTracker.models import build_model
from MAPS.utils import SegmentatorNetwork, UNet, UNetGN


def _disable_require_grad(model):
    for param in model.parameters():
        param.requires_grad = False


class PredictionModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        mask_threshold: float = 0.5,
        input_channels: int = 1,
        loss_weight=None,
        model_type: str = "",
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

        TmpArgs = namedtuple('TmpArgs', ['model_name', 'num_classes', 'selected_loss', 'cat_emb_dim', 'residualUNet', 'device', 'device_id'])
        pre_trained_args = TmpArgs(model_type, 3, defaultdict(str), 4, True, device, 0)
        pretrained = build_model(pre_trained_args)
        
        self.pre_trained = pretrained

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
