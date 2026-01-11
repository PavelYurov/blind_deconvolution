import numpy as np

import torch.nn as nn
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, BatchNorm2d, LeakyReLU, Softmax
import torch.nn.functional as F
import torch
from .model_tools import *


class ActivationNet(nn.Module):
    """
    Takes a QPI measurement and returns the activation map (defect locations and intensity)
    """
    def __init__(self, input_channels=1):
        super().__init__()
        #  These values make sure the image dimensions stay the same
        self.input_channels = input_channels
        self.down1 = Down(2 ** 2, 2 ** 4)
        self.down2 = Down(2 ** 4, 2 ** 6)
        self.up1 = Up(2 ** 6, 2 ** 4)
        self.up2 = Up(2 ** 4, 2 ** 2)
        pad = 2
        ker_same = (5, 5)
        self.encoder_decoder = nn.Sequential(

            # Input layer
            Conv2d(input_channels, 2 ** 2, kernel_size=ker_same, padding=pad, bias=False),
            BatchNorm2d(2 ** 2),
            LeakyReLU(),

            # Hidden layers:

            self.down1,

            self.down2,

            self.up1,

            self.up2,

            # Output layer

            Conv2d(2 ** 2, 1, kernel_size=ker_same, padding=pad, bias=False),
            ReLU()
        )

    def forward(self, x_in):
        x = normalize_tensor_0to1(x_in)
        x = self.encoder_decoder(x)
        x = normalize_tensor_sumto1(x)
        return x


class LISTA(nn.Module):
    """
    This model is the unrolled ISTA algorithm
    """
    def __init__(self, layer_num, iter_num=10):
        super(LISTA, self).__init__()
        self.iter_num = iter_num
        self.layer_num = layer_num
        pad = (2, 2)  # These values make sure the image dimensions stay the same
        ker_same = (5, 5)
        self.x_layers = nn.ModuleList()
        self.y_layers = nn.ModuleList()
        self.slu_layers = nn.ModuleList()
        for _ in range(self.layer_num):
            self.x_layers.append(nn.Conv2d(1, 1, kernel_size=ker_same, padding=pad))
            self.y_layers.append(nn.Conv2d(1, 1, kernel_size=ker_same, padding=pad))
        self.relu = nn.ReLU(inplace=False)

    def set_iter(self, new_iter: int) -> None:
        self.iter_num = new_iter

    def forward(self, x) -> np.ndarray:
        x = normalize_tensor_0to1(x)
        y = torch.clone(x)
        for _ in range(self.iter_num):
            for layer_idx in range(self.layer_num):
                x = self.relu(self.x_layers[layer_idx](x) + self.y_layers[layer_idx](y))
            x = normalize_tensor_sumto1(x)
        return x


# Failed networks:


# class KerNet(nn.Module):
#     """
#     Cnn that shrinks by factor 8 and keeps the channel number the same.
#     """
#
#     def __init__(self):
#         super(KerNet, self).__init__()
#         # Doesn't change the size of the image
#         self.filter_expand = nn.Sequential(
#             DoubleConv(1, 2 ** 6),
#         )
#         # shrinks transverse size by a factor of 2.
#         self.down1 = Down(2 ** 6, 2 ** 7)
#         self.down2 = Down(2 ** 7, 2 ** 8)
#         self.down3 = Down(2 ** 8, 2 ** 9)
#         factor = 2
#         self.up1 = Up(2 ** 9, 2 ** 8)
#         self.up2 = Up(2 ** 8, 2 ** 7)
#         self.out = OutConv(2 ** 7, 1)
#
#     def forward(self, x):
#         expanded = self.filter_expand(x)
#         x = self.down1(expanded)
#         x = self.down2(x)
#         x = self.down3(x)
#         x = self.up1(x)
#         x = self.up2(x)
#         return self.out(x)


# class ActivationResNet(nn.Module):
#     def __init__(self, input_channels=1):
#         super().__init__()
#         self.input_channels = input_channels
#         self.down1 = Down(2 ** 2, 2 ** 3)
#         self.down2 = Down(2 ** 3, 2 ** 4 // 2)
#         self.up1 = UpCombine(2 ** 4, 2 ** 3 // 2)
#         self.up2 = UpCombine(2 ** 3, 2 ** 2)
#         pad = 2  # These values make sure the image dimensions stay the same
#         ker_same = (5, 5)
#         self.input = nn.Sequential(
#             Conv2d(input_channels, 2 ** 2, kernel_size=ker_same, padding=pad, bias=False),
#             BatchNorm2d(2 ** 2),
#             LeakyReLU(),
#         )
#         self.output = nn.Sequential(
#             Conv2d(2 ** 2, 1, kernel_size=ker_same, padding=pad, bias=False),
#             ReLU()
#         )
#
#     def forward(self, x_in):
#         x = (x_in - torch.min(x_in)) / torch.max(x_in)
#         x1 = self.input(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up1(x3, x2)
#         x = self.up2(x, x1)
#         x = self.output(x)
#         x = model_tools.normalize_tensor_sumto1(x)
#         return x


# class ActivationSmiResNet(nn.Module):
#     def __init__(self, input_channels=1):
#         super().__init__()
#         self.input_channels = input_channels
#         self.down1 = Down(2 ** 2, 2 ** 3)
#         self.down2 = Down(2 ** 3, 2 ** 4 // 2)
#         self.up1 = UpCombine(2 ** 4, 2 ** 3)
#         self.up2 = Up(2 ** 3, 2 ** 2)
#         pad = 2  # These values make sure the image dimensions stay the same
#         ker_same = (5, 5)
#         self.input = nn.Sequential(
#             Conv2d(input_channels, 2 ** 2, kernel_size=ker_same, padding=pad, bias=False),
#             BatchNorm2d(2 ** 2),
#             LeakyReLU(),
#         )
#         self.output = nn.Sequential(
#             Conv2d(2 ** 2, 1, kernel_size=ker_same, padding=pad, bias=False),
#             ReLU()
#         )
#
#     def forward(self, x_in):
#         x = (x_in - torch.min(x_in)) / torch.max(x_in)
#         x1 = self.input(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up1(x3, x2)
#         x = self.up2(x)
#         x = self.output(x)
#         x = model_tools.normalize_tensor_sumto1(x)
#         return x
