import torch
from torch import nn
import numpy as np
from torch.nn.modules.loss import _Loss
from typing import Optional
import torch.nn.functional as F
from einops import rearrange

# class DiceLoss:
#     def __call__(self, inputs, targets, smooth=0.0001):
#         # comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = torch.sigmoid(inputs)

#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection) / (inputs.sum() + targets.sum() + smooth)
#         return 1 - dice


class GeneralisedCE(_Loss):
    def __init__(self, q = 0.8, indexToIgnore  = None):
        super().__init__()
        self.q = q
        self.indexToIgnore = indexToIgnore
    
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = rearrange(y_pred, 'b c z h w -> b (z h w) c')
        y_true = rearrange(y_true, 'b c z h w -> b (z h w) c')
        
        y_pred = F.softmax(y_pred, dim=-1)

        # If there are indices to ignore we first replace them with an arbitrary index (0)
        # and later ignore them
        if self.indexToIgnore is not None:
            ignore_index_mask = y_true == self.indexToIgnore
            y_true[ignore_index_mask] = 0

        y_pred = torch.gather(y_pred, 2, y_true)
        loss = (1 - y_pred**self.q) / self.q

        # Now ignore the loss terms for the indices to ignore
        if self.indexToIgnore is not None:
            loss = loss[~ignore_index_mask]
        return loss.mean()


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

    def soft_dice_score(self, output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0,
        eps: float = 1e-7, dims=None,) -> torch.Tensor:
        assert output.size() == target.size()
        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)
            cardinality = torch.sum(output + target, dim=dims)
        else:
            intersection = torch.sum(output * target)
            cardinality = torch.sum(output + target)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        return dice_score



class BaseTripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0, p: int = 2):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=p)

    def calc_features(self, feature_src, true_masks_src):
        # separate bf and classes

        mask_flat = true_masks_src.reshape(1, -1)
        embedding_flat = feature_src.reshape(64, -1)

        pos_0 = torch.where(mask_flat == 0)[1]
        pos_1 = torch.where(mask_flat >= 1)[1]

        if len(pos_0) > 0 and len(pos_1) > 0:
            features_scr_0 = embedding_flat[:, pos_0]
            features_scr_1 = embedding_flat[:, pos_1]

            rand_pos_0 = torch.randint(features_scr_0.shape[1], (1, features_scr_0.shape[1])).long()
            rand_pos_1 = torch.randint(features_scr_1.shape[1], (1, features_scr_1.shape[1])).long()

            features_discrim_0 = torch.transpose(embedding_flat[:, rand_pos_0].squeeze(1), 0, 1)
            features_discrim_1 = torch.transpose(embedding_flat[:, rand_pos_1].squeeze(1), 0, 1)

            return features_discrim_0, features_discrim_1

        return None, None


class BinaryTripletLoss(BaseTripletLoss):
    def __init__(self, margin: float = 1.0, p: int = 2):
        super().__init__(margin, p)

    def forward(self, features_src, true_masks_src):
        features_discrim_0, features_discrim_1 = self.calc_features(features_src, true_masks_src)

        if (features_discrim_0, features_discrim_1) != (None, None):
            feature_number = np.min((int(features_discrim_0.shape[0] / 2), int(features_discrim_1.shape[0] / 2)))
            anchor_0 = features_discrim_0[:feature_number, :]
            positive_0 = features_discrim_0[feature_number:feature_number * 2, :]
            negative_0 = features_discrim_1[:feature_number, :]

            anchor_1 = features_discrim_1[:feature_number, :]
            positive_1 = features_discrim_1[feature_number:feature_number * 2, :]
            negative_1 = features_discrim_0[:feature_number, :]

            loss_triplet = self.loss_fn(anchor_0, positive_0, negative_0) * 0.5 + \
                           self.loss_fn(anchor_1, positive_1, negative_1) * 0.5
            return loss_triplet
        else:
            return 0


class TripletLoss(BaseTripletLoss):
    def __init__(self, margin: float = 1.0, p: int = 2):
        super().__init__(margin, p)

    def forward(self, features_src, true_masks_src):
        features_discrim_0, features_discrim_1 = self.calc_features(features_src, true_masks_src)

        if (features_discrim_0, features_discrim_1) != (None, None):
            feature_number = np.min((int(features_discrim_0.shape[0] / 2), int(features_discrim_1.shape[0] / 2)))
            anchor = features_discrim_0[:feature_number, :]
            positive = features_discrim_0[feature_number:feature_number * 2, :]
            negative = features_discrim_1[:feature_number, :]

            loss_triplet = self.loss_fn(anchor, positive, negative)
            return loss_triplet
        else:
            return 0


class TopologyLoss:
    def __init__(self, device = 'cpu'):
        self.device = device

    def __call__(self, output, target):
        '''
        our modified topology loss
        '''
        # to do -> define the coefficient more elegant way based on hierarchy

        # penalties
        pn = 1  # no penalties
        p1 = 2
        p2 = 4
        p3 = 6
        # manually build based on the classes
        topology_map = [[pn, p3, p3, p3, p3],
                        [p3, pn, p2, p2, p2],
                        [p3, p2, pn, p1, pn],
                        [p3, p2, p1, pn, pn],
                        [p3, p2, pn, pn, pn]]

        # softmax probability
        sm = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(ignore_index=255)

        # probabilities
        probabilities = sm(output)

        target = target.squeeze(1)

        output_squeeze = output
        output_squeeze = output_squeeze.cpu().detach().numpy()
        output_threshold = output_squeeze

        topology_matrix = np.zeros(output_threshold.shape)

        # calculate topology matrix
        for b in range(0, output_threshold.shape[0]):
            for i in range(0, output_threshold.shape[2]):
                for j in range(0, output_threshold.shape[3]):
                    for z in range(0, output_threshold.shape[4]):
                        values = output_threshold[b, :, i, j, z]
                        values_positive = np.argmax(values)
                        top_c = topology_map[values_positive]
                        topology_matrix[b, :, i, j, z] = np.asarray(top_c)

        topology_matrix = torch.tensor(topology_matrix, device=self.device).float()

        probabilities = probabilities * topology_matrix
        top_loss = loss(probabilities, target)

        return top_loss

