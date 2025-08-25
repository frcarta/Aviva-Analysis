import torch.nn as nn
import torch
from math import ceil, floor
import numpy as np
from tools.cross_correlation import xcorr_torch as ccorr
import torch.nn.functional as F

from tools.pytorch_degradation import degrade_image
import math


class SpectralLoss(nn.Module):
    def __init__(self, mtf, ratio, device):

        # Class initialization
        super(SpectralLoss, self).__init__()
        kernel = mtf
        # Parameters definition
        self.nbands = kernel.shape[-1]
        self.device = device
        self.ratio = ratio

        # Conversion of filters in Tensor
        self.pad = floor((kernel.shape[0] - 1) / 2)

        self.cut_border = kernel.shape[0] // 2 // ratio

        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)
        """
        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(
            in_channels=self.nbands,
            out_channels=self.nbands,
            groups=self.nbands,
            kernel_size=kernel.shape,
            bias=False,
            padding="same",  # ADDED
        ) """

        # TODO test
        self.depthconv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            groups=1,
            kernel_size=kernel.shape,
            bias=False,
            padding="same",  # ADDED
        )

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction="mean")

    def forward(self, outputs, labels):
        # outputs = F.pad(
        #     outputs, (self.pad, self.pad, self.pad, self.pad), mode="reflect"
        # )

        degraded_outputs = self.depthconv(outputs)
        # outputs = outputs[:, :, 3 :: self.ratio, 3 :: self.ratio]
        # outputs = outputs[:, :, 1 :: self.ratio, 1 :: self.ratio]
        degraded_outputs = degraded_outputs[:, :, 1 :: self.ratio, 1 :: self.ratio]

        loss_value = self.loss(
            degraded_outputs,
            labels[
                :,
                :,
                :,
                :,
                # self.cut_border : -self.cut_border,
                # self.cut_border : -self.cut_border,
            ],
        )

        return loss_value, degraded_outputs


class MySpectralLoss(nn.Module):
    def __init__(self, ratio, device):

        # Class initialization
        super(MySpectralLoss, self).__init__()
        # kernel = mtf
        # Parameters definition
        self.device = device
        self.ratio = ratio

        # Conversion of filters in Tensor
        sigma = 0.5 * ratio
        radius = int(math.ceil(3 * sigma))
        self.radius = radius

        kernel_size = 2 * radius + 1

        # Create 1D Gaussian kernel
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss /= gauss.sum()

        # Create 2D separable kernel
        kernel2d = gauss[:, None] @ gauss[None, :]
        self.kernel2d = kernel2d.expand(1, 1, kernel_size, kernel_size).to(self.device)

        self.loss = nn.L1Loss(reduction="mean")

    def forward(self, outputs, labels):
        # Pad image
        img_padded = F.pad(
            outputs,
            (self.radius, self.radius, self.radius, self.radius),
            mode="reflect",
        )

        # Convolve with Gaussian
        img_filtered = F.conv2d(img_padded, self.kernel2d)

        # Downsample by factor r
        degraded_outputs = img_filtered[0, :, :: self.ratio, :: self.ratio]

        # outputs = F.pad(
        #     outputs, (self.pad, self.pad, self.pad, self.pad), mode="reflect"
        # )

        # degraded_outputs = self.depthconv(outputs)
        # outputs = outputs[:, :, 3 :: self.ratio, 3 :: self.ratio]
        # outputs = outputs[:, :, 1 :: self.ratio, 1 :: self.ratio]
        # degraded_outputs = degraded_outputs[:, :, 1 :: self.ratio, 1 :: self.ratio]

        loss_value = self.loss(
            degraded_outputs,
            labels[
                :,
                :,
                :,
                :,
                # self.cut_border : -self.cut_border,
                # self.cut_border : -self.cut_border,
            ],
        )

        return loss_value, degraded_outputs


class StructuralLoss(nn.Module):

    def __init__(self, sigma):
        # Class initialization
        super(StructuralLoss, self).__init__()

        # Parameters definition:
        self.scale = ceil(sigma / 2)

    def forward(self, outputs, labels, xcorr_thr):
        x_corr = torch.clamp(ccorr(outputs, labels, self.scale), min=-1)
        x = 1.0 - x_corr

        with torch.no_grad():
            loss_cross_corr_wo_thr = torch.mean(x)

        worst = x.gt(xcorr_thr)
        y = x * worst
        loss_cross_corr = torch.mean(y)

        return loss_cross_corr, loss_cross_corr_wo_thr.item()
