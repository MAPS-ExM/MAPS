import time
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import StepLR
from MAPS.utils import SegmentatorNetwork, UNet


class DiceLoss(_Loss):
    """Taken from https://github.dev/qubvel/segmentation_models.pytorch"""

    def __init__(
        self,
        mode: str,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {"binary", "multilabel", "multiclass"}
        super(DiceLoss, self).__init__()
        self.mode = mode

        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == "multiclass":
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        # breakpoint()
        if self.mode == "binary":
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == "multiclass":
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == "multilabel":
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return self.soft_dice_score(output, target, smooth, eps, dims)

    def soft_dice_score(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims=None,
    ) -> torch.Tensor:
        assert output.size() == target.size()
        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)
            cardinality = torch.sum(output + target, dim=dims)
        else:
            intersection = torch.sum(output * target)
            cardinality = torch.sum(output + target)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        return dice_score


class FineTuneModel(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4, n_classes=5):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_network = UNet(
            encoder_channels=[1, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 32, 1],
            residual=True,
            type="3D",
        )
        self.segmentation_head = SegmentatorNetwork(n_classes, in_classes=32)
        # self.loss = torch.nn.CrossEntropyLoss()
        self.ignore_index = 6
        self.loss = DiceLoss(mode="multiclass", ignore_index=self.ignore_index)
        self.weight_decay = weight_decay
        self.lr = lr
        self.epoch = 0

    def forward(self, x):
        return self.segmentation_head(self.embedding_network(x))

    def comp_loss(self, batch):
        x, y = batch["image"], batch["target"].squeeze(1).long()
        y[y == 5] = 6
        pred = self.segmentation_head(self.embedding_network(x))
        loss = self.loss(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.comp_loss(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        print(time.strftime(f"%d/%m/%Y %H:%M:%S Epoch {self.epoch} finished"))
        self.epoch += 1

    def validation_step(self, batch, batch_idx):
        loss = self.comp_loss(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1, keepdim=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": StepLR(optimizer, step_size=10, gamma=0.1),  # Stepwise scheduler after 5 epochs
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
