import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=False)
        bn_depth = nn.BatchNorm2d(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=False)
        bn_point = nn.BatchNorm2d(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU()),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU())
                                                    ]))

    def forward(self, x):
        return self.block(x)

        
class DecoderSPP(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sep1 = SeparableConv2d(96, 80, relu_first=False)
        self.sep2 = SeparableConv2d(80, 40, relu_first=False)
        self.conv = nn.Conv2d(40, 1, 1, bias=False)



    def forward(self, x):
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        # print(x.shape)
        x = self.sep1(x)
        x = self.sep2(x)
        # print(x.shape)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        # print(x.shape)
        return x