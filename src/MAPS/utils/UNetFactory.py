import torch

from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import Optional, Iterable, Tuple, List

import torch
from torch import nn


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        nn.init.kaiming_uniform_(self.conv.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        return self.conv(x)


class SegmentatorNetwork(nn.Module):
    def __init__(self, n_classes, in_classes=64, bilinear=True):
        super(SegmentatorNetwork, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.outc = OutConv(in_classes, n_classes)

    def forward(self, x):
        out = self.outc(x)
        return out


class CustModule(nn.Module):
    def __init__(self, type: str = "2D"):
        super().__init__()
        self.layers = nn.Identity()
        self.Conv = nn.Conv2d if type == "2D" else nn.Conv3d
        self.BatchNorm = nn.BatchNorm2d if type == "2D" else nn.BatchNorm3d
        self.MaxPool = nn.MaxPool2d if type == "2D" else nn.MaxPool3d
        self.ConvTranspose = nn.ConvTranspose2d if type == "2D" else nn.ConvTranspose3d

    def pool_down_kernel(self, depth_down: bool = True):
        if depth_down:
            return 2
        else:
            return 1, 2, 2

    def convTransParam(self, type="2D", depth_up=True):
        if depth_up:
            return {"kernel_size": 2, "stride": 2}
        else:
            return {"kernel_size": (1, 2, 2), "stride": (1, 2, 2)}

    def forward(self, x, cat_embedding=None):
        if cat_embedding is None:
            return self.layers(x)
        else:
            return self.layers(x, cat_embedding)


def UNetFactory(ConvLayer):
    class DownLayer(CustModule):
        """
        Downsamples the input by a factor of 2 and processes it with a Conv Layer
        """

        def __init__(
            self,
            in_c: int,
            out_c: int,
            type: int = "2D",
            depth_downsample: bool = True,
            residual=False,
            cat_emb_dim=None,
        ):
            super().__init__(type)
            self.down_pool = self.MaxPool(kernel_size=self.pool_down_kernel(depth_downsample))
            self.conv_layer = ConvLayer(in_c, out_c, type, residual=residual, cat_emb_dim=cat_emb_dim)

        def forward(self, x, cat_embedding=None):
            return self.conv_layer(self.down_pool(x), cat_embedding)

    class Encoder(CustModule):
        def __init__(self, channels, type="2D", depth_downsample=None, residual=False, cat_emb_dim=None):
            super().__init__(type)
            self.channels = channels
            self.depth_downsample = (
                depth_downsample if (type == "3D" and depth_downsample is not None) else [True] * (len(channels) - 2)
            )
            self.layers = nn.ModuleDict(
                {
                    # TODO: Is this a bug that I do use the OGConvLayer here with batch norm or do I want this?
                    "Down0": OGConvLayer(
                        in_c=self.channels[0],
                        out_c=self.channels[1],
                        type=type,
                        first_kernel=7,
                        first_stride=(1, 2, 2),
                        first_padding=3,
                        cat_emb_dim=cat_emb_dim,
                    )
                }
            )

            # Shift channels by one and zip to generate down-sampling path
            for i, (in_c, out_c, depth_down) in enumerate(
                zip(self.channels[1:], self.channels[2:], self.depth_downsample)
            ):
                self.layers[f"Down{i + 1}"] = DownLayer(
                    in_c, out_c, type, depth_down, residual=residual, cat_emb_dim=cat_emb_dim
                )

        def forward(self, x, cat_embedding=None):
            features = []
            for l in self.layers.values():
                x = l(x, cat_embedding)
                features.append(x)
            return features

    class UpLayer(CustModule):
        """
        First upsamples the input by a factor of 2, concats the skip-connection and outputs it through another Conv-Layer
        """

        def __init__(
            self,
            in_c: int,
            concat_c: int,
            out_c: int,
            type: str = "2D",
            depth_upsample: bool = True,
            interpolate: bool = True,
            dropout: bool = True,
            convTransParam: dict = None,
            residual=False,
            cat_emb_dim: Optional[int] = None,
        ):
            super().__init__(type)
            if convTransParam is None:
                convTransParam = self.convTransParam(type, depth_upsample)
            self.dropout = nn.Dropout(p=0.25) if dropout else nn.Identity()
            if interpolate:
                self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
                self.conv_layer = ConvLayer(in_c + concat_c, out_c, type, residual=residual, cat_emb_dim=cat_emb_dim)
            else:
                self.upsample = self.ConvTranspose(in_c, out_c, **convTransParam)
                self.conv_layer = ConvLayer(out_c + concat_c, out_c, type, residual=residual, cat_emb_dim=cat_emb_dim)

        def forward(self, x, skip=None, cat_embedding=None):
            # print(f"{'Up:':7} Input shape: {str(list(x.shape)):22} + {str(list(skip.shape)):22} ")
            x = self.upsample(x)
            # print(f"{'After:':7} Input shape: {str(list(x.shape)):22} + {str(list(skip.shape)):22} ")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            return self.conv_layer(self.dropout(x), cat_embedding)

    class Decoder(CustModule):
        def __init__(
            self,
            channels: Iterable[int],
            enc_channels: Iterable[int],
            type: str = "2D",
            depth_upsample: Optional[Iterable[int]] = None,
            interpolate=False,
            dropout: bool = True,
            residual=False,
            cat_emb_dim: Optional[int] = None,
        ) -> None:
            super().__init__(type)
            assert channels[0] == enc_channels[-1], (
                "Decoder has to start with the same number of channels as encoder ends"
            )
            self.channels = channels
            # self.enc_channels = enc_channels[-2:0:-1]  # Reverse and exclude the first entry and last
            self.enc_channels = enc_channels[-2::-1]  # Reverse and exclude the last entry
            self.depth_upsample = (
                depth_upsample[::-1] if (type == "3D" and depth_upsample is not None) else [True] * (len(channels) - 1)
            )

            self.layers = nn.ModuleDict({})
            for i, (in_c, enc_c, out_c, d_upsample) in enumerate(
                zip(self.channels, self.enc_channels, self.channels[1:], self.depth_upsample)
            ):
                if i < len(self.channels) - 2:
                    self.layers[f"Up{i}"] = UpLayer(
                        in_c=in_c,
                        concat_c=enc_c,
                        out_c=out_c,
                        type=type,
                        depth_upsample=d_upsample,
                        interpolate=interpolate,
                        dropout=dropout,
                        residual=residual,
                        cat_emb_dim=cat_emb_dim,
                    )
                else:
                    self.layers[f"Up{i}"] = UpLayer(
                        in_c=in_c,
                        concat_c=enc_c,
                        out_c=out_c,
                        type=type,
                        depth_upsample=d_upsample,
                        interpolate=interpolate,
                        dropout=dropout,
                        convTransParam={"kernel_size": (1, 2, 2), "stride": (1, 2, 2)},
                        residual=residual,
                        cat_emb_dim=cat_emb_dim,
                    )

        def forward(self, features: torch.Tensor, cat_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Go backwards through the features and make use of the UpLayers which first upsample and then
            concatenate the skip-connections before another convolution
            """
            x = features[-1]
            for layer, feat in zip(self.layers.values(), features[-2::-1]):
                x = layer(x, feat, cat_embedding)
            return x

    class UNet(nn.Module):
        """
        Decoder- and Encoderchannels should have the same length
        Decoder has to start with the same number of channels as encoder ends
        """

        def __init__(
            self,
            encoder_channels: Iterable[int],
            decoder_channels: Iterable[int],
            type: str = "3D",
            depth_downsample: Optional[Iterable[int]] = None,
            interpolate: bool = False,
            dropout: bool = True,
            residual=False,
            cat_emb_dim: Optional[int] = None,
            device: Optional[str] = None,
        ) -> None:
            """
            This UNet has a ResNet like first layer, that already downsamples by having a stride of 2 in the first layer.
            """
            super().__init__()
            self.depth_downsampling = depth_downsample

            # Check input
            self._check_args(encoder_channels, decoder_channels, type, depth_downsample)

            # Build model
            self.output_dim = decoder_channels[-1]
            self.encoder = Encoder(encoder_channels, type, depth_downsample, residual=residual, cat_emb_dim=cat_emb_dim)
            self.decoder = Decoder(
                decoder_channels[: (len(encoder_channels))],
                encoder_channels,
                type,
                depth_downsample,
                interpolate,
                dropout,
                residual=residual,
                cat_emb_dim=cat_emb_dim,
            )
            # Use the layers not used in the U-architecture for the final layers
            # self.final = HeadLayer(channels=decoder_channels[(len(encoder_channels) - 1):], include_sig=False, type=type)

            # Set device
            self.device = device
            if device is not None:
                self.to(self.device)

        def forward(self, x, cat_embedding=None):
            features = [x] + self.encoder(x, cat_embedding)
            features = self.decoder(features, cat_embedding)
            return features  # self.final(features)

        def comp_features(self, x, cat_embedding=None):
            features = self.encoder(x, cat_embedding)
            features = self.decoder(features, cat_embedding)
            return features

        def predict(self, x, cat_embedding=None):
            res = self.forward(x, cat_embedding)
            return torch.argmax(res, dim=1)

        def _check_args(self, encoder_channels, decoder_channels, type, depth_downsample):
            assert len(encoder_channels) <= len(decoder_channels), "Decoder needs to be longer than encoder"
            assert decoder_channels[0] == encoder_channels[-1], (
                f"Decoder has to start with the same number of channels as encoder ends: {decoder_channels[0]} vs {encoder_channels[-1]}"
            )

            assert type in ["2D", "3D"], "Type has to be either 2D or 3D"
            if type == "2D":
                assert depth_downsample is None, "If type is 2D, there is no depth downsampling possibility!"
            if depth_downsample is not None:
                assert len(depth_downsample) == len(encoder_channels) - 2

    return UNet


class OGConvLayer(CustModule):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        type: str = "2D",
        first_kernel=3,
        first_stride=1,
        first_padding=1,
        residual=False,
        cat_emb_dim=None,
    ):
        super().__init__(type)
        self.conv1 = self.Conv(
            in_c, out_c, kernel_size=first_kernel, stride=first_stride, padding=first_padding, bias=False
        )
        self.norm1 = self.BatchNorm(out_c, momentum=0.05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.Conv(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = self.BatchNorm(out_c, momentum=0.05)
        self.residual = residual

        nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

        # Category embedding
        self.mlp_cat = nn.Linear(cat_emb_dim, 2 * out_c) if cat_emb_dim is not None else None
        if self.mlp_cat is not None:
            nn.init.constant_(self.mlp_cat.weight, 0)
            nn.init.constant_(self.mlp_cat.bias, 0)

    def forward(self, x, cat_embedding=None):
        # Transformh the categorical embedding into scale and shift
        if cat_embedding is not None:
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #     with record_function("model_inference"):
            cat_embedding = self.mlp_cat(cat_embedding)
            cat_embedding = rearrange(cat_embedding, "b c -> b c 1 1 1")  # Just broadcast over all spatial dimensions
            scale_shift = cat_embedding.chunk(2, dim=1)
        else:
            scale_shift = None

        x = self.relu(self.add_category_embedding(self.norm1(self.conv1(x)), scale_shift))
        y = self.relu(self.add_category_embedding(self.norm2(self.conv2(x)), scale_shift))
        if self.residual:
            return x + y
        else:
            return y

    def add_category_embedding(self, x, scale_shift=None):
        if scale_shift is not None:
            return x * (1 + scale_shift[0]) + scale_shift[1]
        else:
            return x


class ConvNeXtLayer(CustModule):
    def __init__(
        self, in_c: int, out_c: int, type: str = "2D", first_kernel=3, first_stride=1, first_padding=1, residual=False
    ):
        super().__init__(type)
        self.conv1 = self.Conv(in_c, in_c, kernel_size=7, stride=1, padding=3, groups=in_c)
        self.bn = nn.BatchNorm3d(in_c, momentum=0.01)
        self.conv2 = self.Conv(in_c, 4 * in_c, kernel_size=1, stride=1, padding=0)
        self.gelu = nn.GELU()
        self.conv3 = self.Conv(4 * in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.convres = self.Conv(in_c, out_c, kernel_size=1, stride=1, padding=0)

        nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        return x + self.convres(residual)


class ConvWN(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvWN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
            .mean(dim=4, keepdim=True)
        )
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class GNConvLayer(OGConvLayer):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        type: str = "2D",
        first_kernel=3,
        first_stride=1,
        first_padding=1,
        residual=False,
        cat_emb_dim=None,
    ):
        super().__init__(in_c=in_c, out_c=out_c)
        self.conv1 = ConvWN(in_c, out_c, kernel_size=first_kernel, stride=first_stride, padding=first_padding)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvWN(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_c)
        self.residual = residual

        nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_in", nonlinearity="relu")

        # Category embedding
        self.mlp_cat = nn.Sequential(nn.Linear(cat_emb_dim, 2 * out_c)) if cat_emb_dim is not None else None

        if self.mlp_cat is not None:
            nn.init.constant_(self.mlp_cat[0].weight, 0)
            nn.init.constant_(self.mlp_cat[0].bias, 0)


UNet = UNetFactory(OGConvLayer)
UNeXt = UNetFactory(ConvNeXtLayer)
UNetGN = UNetFactory(GNConvLayer)

if __name__ == "__main__":
    input_channels = 1
    net = UNet(
        encoder_channels=[input_channels, 64, 128, 256, 512],
        decoder_channels=[512, 256, 128, 64, 32, 1],
        type="3D",
        cat_emb_dim=32,
    ).cuda()

    x = torch.randn(4, 1, 16, 128, 128).cuda()
    pred = net(x)

    emb = torch.randn(4, 32).cuda()
    pred = net(x, emb)
    print("Finished")
