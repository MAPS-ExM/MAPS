import os
import sys
import time
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import tifffile
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch import optim

from models.DiceLoss import DiceLoss
from models.predict import predict_stack

from utils.UNetFactory import UNetGN


class OutlineModel(pl.LightningModule):
    def __init__(
        self,
        train_step_max_thres=50000,
        train_steps_warm_up=500,
        threshold_upper=0.25,
        threshold_lower=0.25,
        label_smoothing=0.1,
        pos_weight=1,
        loss='BCEWithLogitsLoss',
        lr=1e-3,
        L2=1e-4,
    ):
        super().__init__()
        self.model = UNetGN(
            encoder_channels=[1, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 1],
            residual=True,
            type='3D',
            cat_emb_dim=None,
        )
        self.save_hyperparameters()
        self.lr = lr
        self.L2 = L2
        self.seg_head = torch.nn.Conv3d(32, 1, kernel_size=1, padding=0)
        self.loss_name = loss
        if loss == 'BCEWithLogitsLoss':
            self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        elif loss == 'Dice':
            self.index_to_ignore = 3
            self.loss = DiceLoss(mode='binary', ignore_index=self.index_to_ignore, from_logits=True)
            assert label_smoothing == 0, 'Label smoothing not implemented for Dice loss'
        else:
            raise NotImplementedError(f'Loss {loss} not implemented')
        # self.loss = torch.nn.BCELoss()
        self.losses = []
        self.train_step = 0
        self.train_step_max_thres = train_step_max_thres
        self.train_steps_warm_up = train_steps_warm_up
        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.n_added_pix = []
        self.n_ignored_antibodies = []
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

        self.validation_step_outputs = []

    def on_train_epoch_start(self) -> None:
        print(f'{time.strftime("%H:%M:%S")} Epoch {self.trainer.current_epoch}')

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        # We start with all antibodies as targets (i.e not masked 0)
        mask[y > 0] = 0
        logit = self.seg_head(self.model(x))
        pred = F.sigmoid(logit)

        # Select pixels from the masked area that we include as positive examples (y=1 and mask=0)
        if self.train_step > self.train_steps_warm_up:
            threshold_upper = 1 - self.threshold_upper * min(
                1,
                self.train_step / self.train_step_max_thres,
            )
            indices = torch.logical_and(
                torch.logical_and(pred > threshold_upper, mask == 1),  # High confidence and masked
                y == 0,
            )  # ... but currently not marked with antibodies (only important for reporting)
            # Report
            n_added_pixel = indices.sum().detach().cpu().item()
            n_candiate_pixel = max(torch.logical_and(mask == 1, y == 0).sum().detach().cpu().item(), 1)
            self.n_added_pix.append(n_added_pixel / n_candiate_pixel)
            # Modify the mask accordingly
            mask[indices] = 0  # We don't mask these pixels any longer
            y[indices] = 1  # ... but include them as postive targets

            # Select antibodies that we ignore
            threshold_lower = self.threshold_lower * min(
                1,
                self.train_step / self.train_step_max_thres,
            )
            indices = torch.logical_and(pred < threshold_lower, y == 1)
            # y[indices] = 0  # Not needed because we don't want to label them negative but just start to ignore them
            mask[indices] = 1  # We ignore these pixels in the loss calculation
            n_ignored_anti = indices.sum().detach().cpu().item()
            n_candiate_pixel = max((y == 1).sum().detach().cpu().item(), 1)
            self.n_ignored_antibodies.append(n_ignored_anti / n_candiate_pixel)
        else:
            self.n_added_pix.append(0.0)
            self.n_ignored_antibodies.append(0.0)
            threshold_upper = 1.0
            threshold_lower = 0.0

        loss = self.calc_loss(logit, y, mask)

        self.losses.append(loss.detach().cpu().item())
        self.log('train_loss', loss)
        self.log('threshold_upper', threshold_upper)
        self.log('threshold_lower', threshold_lower)
        try:
            self.log('mean_prob_mask', pred[mask > 0].detach().mean().cpu().item())
        except Exception:
            self.log('mean_prob_mask', 0.0)
        self.log('mean_prob_outisde_mask', pred[mask == 0].detach().mean().cpu().item())

        try:
            # self.log('n_added_pixel', n_added_pixel / (mask == 0).sum().detach().cpu().item())
            self.log('n_added_pixel', self.n_added_pix[-1])
        except ZeroDivisionError:
            self.log('n_added_pixel', 0.0)
        try:
            # self.log('n_ignored_anti', n_ignored_anti / (y == 1).sum().detach().cpu().item())
            self.log('n_ignored_anti', self.n_ignored_antibodies[-1])
        except ZeroDivisionError:
            self.log('n_ignored_anti', 0.0)
        self.train_step += 1
        return loss

    def calc_loss(self, logit, y, mask):
        # Mask == 1 means we want to exclude those pixels from the loss
        if self.loss_name == 'Dice':
            y[mask > 0] = self.index_to_ignore
            loss = self.loss(logit, y)
        elif self.loss_name == 'BCEWithLogitsLoss':
            if self.label_smoothing > 0:
                y_smooth = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            loss = self.loss(logit[mask == 0], y_smooth[mask == 0])
        else:
            raise NotImplementedError(f'Loss {self.loss_name} not implemented')
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        logit = self.seg_head(self.model(x))
        pred = F.sigmoid(logit)
        loss = self.calc_loss(logit, y, mask)
        self.validation_step_outputs.append(
            (
                pred[mask > 0].detach().cpu().numpy(),
                pred[mask == 0].detach().cpu().numpy(),
            )
        )
        self.log('val_loss', loss)
        return loss

    def on_validation_end(self):
        predictions_loss = np.concatenate([x[0] for x in self.validation_step_outputs])
        predictions_ambig = np.concatenate([x[1] for x in self.validation_step_outputs])
        self.logger.experiment.add_histogram('Pred Prob Loss', predictions_loss, global_step=self.trainer.current_epoch)
        self.logger.experiment.add_histogram(
            'Pred Prob Ambiguous',
            predictions_ambig,
            global_step=self.trainer.current_epoch,
        )
        self.validation_step_outputs.clear()

    def forward(self, x):
        return self.seg_head(self.model(x))

    def predict(self, x):
        return (self.forward(x) > 0).float()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.L2)
        return optimizer

    def make_pred(self, nhs, antibody, antibody_dil, test_ind):
        with torch.no_grad():
            model = self.to('cuda')
            pred = model(nhs[test_ind, None].float().cuda()).cpu().numpy()[0, 0]

        fig, ax = plt.subplots(1, 3, figsize=(14, 5))
        ax[0].imshow(nhs[test_ind], cmap='gray', interpolation='none')
        ax[0].imshow(
            antibody[test_ind] > 0,
            cmap=ListedColormap(['none', 'Red']),
            interpolation='none',
            alpha=0.8,
        )
        ax[1].imshow(pred, cmap='gray', interpolation='none')
        ax[1].set_title('Prediction')
        ax[2].imshow(nhs[test_ind], cmap='gray', interpolation='none')
        mask = np.logical_or(antibody_dil[test_ind] == 0, antibody[test_ind] > 0)
        ax[2].imshow(
            mask == 0,
            cmap=ListedColormap(['none', 'Blue']),
            interpolation='none',
            alpha=0.5,
        )
        ax[2].imshow(
            antibody[test_ind] > 0,
            cmap=ListedColormap(['none', 'Red']),
            interpolation='none',
            alpha=0.8,
        )
        ax[2].set_title('Antibodies')
        plt.savefig(os.path.join(self.logger.log_dir, 'test_prediction.png'))

    def make_full_pred(self, nhs, file_name, include_prob=False, path=None):
        if path is None:
            path = self.logger.log_dir
        print(f'{time.strftime("%H:%M:%S")} Predicting {file_name}')
        with torch.no_grad():
            model = self.to('cuda')
            if nhs.__class__.__name__ == 'ndarray':
                nhs = torch.from_numpy(nhs)
            test = predict_stack(
                nhs.float().cuda(),
                model,
                img_window=(32, 256, 256),
                stride=(16, 128, 128),
            )
            # try:
            #     test = model(torch.from_numpy(nhs[None, None]).float().cuda()).cpu().numpy().squeeze()
            # except RuntimeError:
            #     model = model.cpu()
            #     max_z = 2**int(np.floor(np.log2(nhs.shape[0])))
            #     print('Shape: ', nhs.shape, max_z)
            #     test = model(torch.from_numpy(nhs[:max_z][None, None]).float()).cpu().numpy().squeeze()
        if include_prob:
            tifffile.imwrite(
                os.path.join(path, f'{file_name}_prob.tif'),
                test,
                imagej=True,
                metadata={'axes': 'ZYX'},
                compression='zlib',
            )

        for ind in np.linspace(10, nhs.shape[0] - 10, num=4).astype('int'):
            fig, ax = plt.subplots(1, 2, figsize=(14, 10))
            ax[0].imshow(nhs[ind], cmap='gray', interpolation='none')
            ax[1].imshow(nhs[ind], cmap='gray', interpolation='none')
            ax[1].imshow(
                test[ind] > 0,
                cmap=ListedColormap(['none', 'Green']),
                interpolation='none',
                alpha=0.8,
            )
            ax[1].set_title('Prediction')
            plt.savefig(os.path.join(path, f'{file_name}_ind_{ind}bin.png'))

        tifffile.imwrite(
            os.path.join(path, f'{file_name}_bin.tif'),
            (test > 0).astype('uint8'),
            imagej=True,
            metadata={'axes': 'ZYX'},
            compression='zlib',
        )


class InitBoundaryNet(OutlineModel):
    def training_step(self, batch, batch_idx):
        # Different from the UNet, we ignore everything outside of the mask
        x, mask, y = batch
        logit = self.seg_head(self.model(x))
        pred = F.sigmoid(logit)

        # Select pixels from the masked area that we include as positive examples (y=1 and mask=0)
        if self.train_step > self.train_steps_warm_up:
            threshold_upper = 1 - self.threshold_upper * min(
                1,
                self.train_step / self.train_step_max_thres,
            )
            indices = torch.logical_and(pred > threshold_upper, mask == 0)  # High confidence and masked

            # Report
            n_added_pixel = indices.sum().detach().cpu().item()
            n_candiate_pixel = min(torch.logical_and(mask == 1, y == 0).sum().detach().cpu().item(), 1)
            self.n_added_pix.append(n_added_pixel / n_candiate_pixel)
            # Modify the mask accordingly
            mask[indices] = 1  # We don't mask these pixels any longer
            y[indices] = 1  # ... but include them as postive targets

            # Select antibodies that we ignore
            threshold_lower = self.threshold_lower * min(
                1,
                self.train_step / self.train_step_max_thres,
            )
            indices = torch.logical_and(pred < threshold_lower, y == 1)
            # y[indices] = 0  # Not needed because we don't want to label them negative but just start to ignore them
            mask[indices] = 0  # We ignore these pixels in the loss calculation
            n_ignored_anti = indices.sum().detach().cpu().item()
            n_candiate_pixel = min((y == 1).sum().detach().cpu().item(), 1)
            self.n_ignored_antibodies.append(n_ignored_anti / n_candiate_pixel)
        else:
            self.n_added_pix.append(0.0)
            self.n_ignored_antibodies.append(0.0)
            threshold_upper = 1.0
            threshold_lower = 0.0

        loss = self.calc_loss(logit, y, mask)

        self.losses.append(loss.detach().cpu().item())
        self.log('train_loss', loss)
        self.log('threshold_upper', threshold_upper)
        self.log('threshold_lower', threshold_lower)
        try:
            self.log('mean_prob_mask', pred[mask > 0].detach().mean().cpu().item())
        except Exception:
            self.log('mean_prob_mask', 0.0)
        self.log('mean_prob_outisde_mask', pred[mask == 0].detach().mean().cpu().item())

        try:
            # self.log('n_added_pixel', n_added_pixel / (mask == 0).sum().detach().cpu().item())
            self.log('n_added_pixel', self.n_added_pix[-1])
        except ZeroDivisionError:
            self.log('n_added_pixel', 0.0)
        try:
            # self.log('n_ignored_anti', n_ignored_anti / (y == 1).sum().detach().cpu().item())
            self.log('n_ignored_anti', self.n_ignored_antibodies[-1])
        except ZeroDivisionError:
            self.log('n_ignored_anti', 0.0)
        self.train_step += 1
        return loss

    def calc_loss(self, logit, y, mask):
        # Mask == 1 means we want to exclude those pixels from the loss
        if self.loss_name == 'Dice':
            y[mask == 0] = self.index_to_ignore
            loss = self.loss(logit, y)
        elif self.loss_name == 'BCEWithLogitsLoss':
            if self.label_smoothing > 0:
                y_smooth = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            loss = self.loss(logit[mask > 0], y_smooth[mask > 0])
        else:
            raise NotImplementedError(f'Loss {self.loss_name} not implemented')
        return loss


class BoudnaryUNet(pl.LightningModule):
    def __init__(
        self,
        label_smoothing=0.1,
        pos_weight=1,
        loss='BCEWithLogitsLoss',
        lr=1e-3,
        L2=1e-4,
        train_steps_warm_up=1e9,
        threshold_upper=0.25,
        train_step_max_thres=1e9,
        verbose=False,
    ):
        super().__init__()
        self.model = UNetGN(
            encoder_channels=[1, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 1],
            residual=True,
            type='3D',
            cat_emb_dim=None,
        )
        self.save_hyperparameters()
        self.verbose = verbose
        self.lr = lr
        self.L2 = L2
        self.seg_head = torch.nn.Conv3d(32, 1, kernel_size=1, padding=0)
        self.loss_name = loss
        if loss == 'BCEWithLogitsLoss':
            self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        elif loss == 'Dice':
            self.index_to_ignore = 3
            self.loss = DiceLoss(mode='binary', ignore_index=self.index_to_ignore, from_logits=True)
            assert label_smoothing == 0, 'Label smoothing not implemented for Dice loss'
        else:
            raise NotImplementedError(f'Loss {loss} not implemented')
        # self.loss = torch.nn.BCELoss()
        self.losses = []
        self.train_step = 0
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        self.train_steps_warm_up = train_steps_warm_up
        self.threshold_upper = threshold_upper
        self.train_step_max_thres = train_step_max_thres
        self.n_added_pix = []

    def on_train_epoch_start(self) -> None:
        print(f'{time.strftime("%H:%M:%S")} Epoch {self.trainer.current_epoch}')

    def training_step(self, batch, batch_idx):
        x, y, mask = batch  # Mask: 0 background, 1 outline, 2 dilated antibodies
        # We start with all antibodies as targets

        logit = self.seg_head(self.model(x))
        pred = F.sigmoid(logit)

        # Select pixels from the masked area that we include as positive examples (y=1 and mask=0)
        if self.train_step > self.train_steps_warm_up:
            threshold_upper = 1 - self.threshold_upper * min(
                1,
                self.train_step / self.train_step_max_thres,
            )
            indices = torch.logical_and(
                torch.logical_and(
                    pred > threshold_upper, mask == 2
                ),  # High confidence and masked because dilated antibodies
                y == 0,
            )  # ... but currently not marked with antibodies (only important for reporting)
            # Report
            n_added_pixel = indices.sum().detach().cpu().item()
            n_candiate_pixel = min(torch.logical_and(mask == 1, y == 0).sum().detach().cpu().item(), 1)
            self.n_added_pix.append(n_added_pixel / n_candiate_pixel)
            # Modify the mask accordingly
            mask[indices] = 1  # We don't mask these pixels any longer
            y[indices] = 1  # ... but include them as postive targets

        else:
            self.n_added_pix.append(0.0)
            threshold_upper = 1.0

        loss = self.calc_loss(logit, y, mask)

        self.losses.append(loss.detach().cpu().item())
        if self.verbose:
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        else:
            self.log('train_loss', loss)
        self.train_step += 1
        return loss

    def calc_loss(self, logit, y, mask):
        # Mask != 1 means we want to exclude those pixels from the loss
        if self.loss_name == 'Dice':
            y[mask != 1] = self.index_to_ignore
            loss = self.loss(logit, y)
        elif self.loss_name == 'BCEWithLogitsLoss':
            loss = self.loss(logit[mask == 1], y[mask == 1])
        else:
            raise NotImplementedError(f'Loss {self.loss_name} not implemented')
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        logit = self.seg_head(self.model(x))
        pred = F.sigmoid(logit)
        loss = self.calc_loss(logit, y, mask)
        self.validation_step_outputs.append(
            (
                pred[mask > 0].detach().cpu().numpy(),
                pred[mask == 0].detach().cpu().numpy(),
            )
        )
        self.log('val_loss', loss)
        return loss

    def forward(self, x):
        return self.seg_head(self.model(x))

    def predict(self, x):
        return (self.forward(x) > 0).float()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.L2)
        return optimizer

    def make_pred(self, nhs, antibody, antibody_dil, test_ind):
        with torch.no_grad():
            model = self.to('cuda')
            pred = model(nhs[test_ind, None].float().cuda()).cpu().numpy()[0, 0]

        fig, ax = plt.subplots(1, 3, figsize=(14, 5))
        ax[0].imshow(nhs[test_ind], cmap='gray', interpolation='none')
        ax[0].imshow(
            antibody[test_ind] > 0,
            cmap=ListedColormap(['none', 'Red']),
            interpolation='none',
            alpha=0.8,
        )
        ax[1].imshow(pred, cmap='gray', interpolation='none')
        ax[1].set_title('Prediction')
        ax[2].imshow(nhs[test_ind], cmap='gray', interpolation='none')
        mask = np.logical_or(antibody_dil[test_ind] == 0, antibody[test_ind] > 0)
        ax[2].imshow(
            mask == 0,
            cmap=ListedColormap(['none', 'Blue']),
            interpolation='none',
            alpha=0.5,
        )
        ax[2].imshow(
            antibody[test_ind] > 0,
            cmap=ListedColormap(['none', 'Red']),
            interpolation='none',
            alpha=0.8,
        )
        ax[2].set_title('Antibodies')
        plt.savefig(os.path.join(self.logger.log_dir, 'test_prediction.png'))

    def make_full_pred(self, nhs, file_name, include_prob=False, path=None):
        if path is None:
            path = self.logger.log_dir
        print(f'{time.strftime("%H:%M:%S")} Predicting {file_name}')
        with torch.no_grad():
            model = self.to('cuda')
            if nhs.__class__.__name__ == 'ndarray':
                nhs = torch.from_numpy(nhs)
            test = predict_stack(
                nhs.float().cuda(),
                model,
                img_window=(32, 256, 256),
                stride=(16, 128, 128),
            )
            # try:
            #     test = model(torch.from_numpy(nhs[None, None]).float().cuda()).cpu().numpy().squeeze()
            # except RuntimeError:
            #     model = model.cpu()
            #     max_z = 2**int(np.floor(np.log2(nhs.shape[0])))
            #     print('Shape: ', nhs.shape, max_z)
            #     test = model(torch.from_numpy(nhs[:max_z][None, None]).float()).cpu().numpy().squeeze()
        if include_prob:
            tifffile.imwrite(
                os.path.join(path, f'{file_name}_prob.tif'),
                test,
                imagej=True,
                metadata={'axes': 'ZYX'},
                compression='zlib',
            )

        for ind in np.linspace(10, nhs.shape[0] - 10, num=4).astype('int'):
            fig, ax = plt.subplots(1, 2, figsize=(14, 10))
            ax[0].imshow(nhs[ind], cmap='gray', interpolation='none')
            ax[1].imshow(nhs[ind], cmap='gray', interpolation='none')
            ax[1].imshow(
                test[ind] > 0,
                cmap=ListedColormap(['none', 'Green']),
                interpolation='none',
                alpha=0.8,
            )
            ax[1].set_title('Prediction')
            plt.savefig(os.path.join(path, f'{file_name}_ind_{ind}bin.png'))

        tifffile.imwrite(
            os.path.join(path, f'{file_name}_bin.tif'),
            (test > 0).astype('uint8'),
            imagej=True,
            metadata={'axes': 'ZYX'},
            compression='zlib',
        )
