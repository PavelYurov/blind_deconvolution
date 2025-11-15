import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def tensor_to_nparray(tensor):
    return np.array(tensor.squeeze().cpu().detach().numpy())


def plot_conv(kernel_in, activation_in, target_in):
    A_conv_K = tensor_to_nparray(F.conv2d(activation_in, kernel_in, padding='same'))
    activation = tensor_to_nparray(activation_in)
    kernel = tensor_to_nparray(kernel_in)
    target = tensor_to_nparray(target_in)

    fig, ax = plt.subplots(1, 4, figsize=(9, 3), dpi=150)
    ax[0].set_title('Kernel', fontsize=12)
    ax[0].imshow(kernel, cmap='hot')

    ax[1].set_title('Activation Pred', fontsize=12)
    ax[1].imshow(activation, cmap='hot')

    ax[2].set_title('Convolution', fontsize=12)
    ax[2].imshow(A_conv_K, cmap='hot')

    ax[3].set_title('Measurement', fontsize=12)
    ax[3].imshow(target, cmap='hot')

    for i in range(4):
        ax[i].set_axis_off()
    plt.show()


def smooth_abs(x, eps=1e-18):
    return (x ** 2 + eps) ** 0.5 - (eps ** 0.5)


def normalize_tensor_0to1(t):
    """
    Normalizes a tensor such that the data in each sample in the batch will distribute between 0 and 1.
    """
    shape = t.size()
    t = t.view(t.size(0), -1)
    t -= t.min(1, keepdim=True)[0]
    t /= t.max(1, keepdim=True)[0]
    return t.view(shape)


def normalize_tensor_sumto1(t: torch.Tensor):
    """
    Normalizes a tensor such that the data in each sample in the batch will sum to 1.
    """
    norm = t.sum(dim=1, keepdim=True)
    norm[norm == 0] = 1
    # shape = t.size()
    # t_out = t.view(t.size(0), -1)
    # sum_tensor = t_out.sum(1, keepdim=True)
    # sum_tensor[sum_tensor == 0] = 1
    # t_out /= sum_tensor
    return t/norm


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) *
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class UpCombine(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ActivationLoss(nn.Module):
    def __init__(self, r=0.1):
        super().__init__()
        self.r = r
        self.mu = 10 ** -6

    def regulator(self, activation):
        return torch.sum(self.mu ** 2 * (torch.sqrt(1 + (self.mu ** -2) * torch.abs(activation)) - 1))

    def forward(self, activation_pred, activation, kernel, target=None):
        conv_pred = []
        conv_target = []
        for i in range(activation_pred.shape[0]):
            single_kernel = kernel[i].unsqueeze(dim=0)
            single_activation_pred = activation_pred[i].unsqueeze(dim=0)
            single_activation = activation[i].unsqueeze(dim=0).unsqueeze(dim=0)
            # plot_conv(single_kernel, single_activation, target[i])
            conv_pred.append(F.conv2d(single_activation_pred, single_kernel, padding='same'))
            conv_target.append(F.conv2d(single_activation, single_kernel, padding='same'))
        conv_pred_stack = torch.stack(conv_pred, dim=0).squeeze(dim=1)
        conv_target_stack = torch.stack(conv_target, dim=0).squeeze(dim=1)
        regulation_term = smooth_abs(self.regulator(activation_pred) - self.regulator(activation))

        loss = F.huber_loss(conv_pred_stack, conv_target_stack) / 2
        loss += self.r * regulation_term
        # loss += (self.r * 10 ** - 4) * torch.abs(torch.count_nonzero(activation) - torch.count_nonzero(activation_pred)) / torch.count_nonzero(activation)
        loss /= activation.shape[0]
        if torch.cuda.is_available():
            return loss.cuda()
        return loss


class KernelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, kernel_pred, activation, kernel, target):
        conv_pred = []
        conv_target = []
        for i in range(kernel_pred.shape[0]):
            single_kernel = kernel[i].unsqueeze(dim=0)
            single_kernel_pred = kernel_pred[i].unsqueeze(dim=0)
            single_activation = activation[i].unsqueeze(dim=0).unsqueeze(dim=0)
            # plot_conv(single_kernel, single_activation, target[i])
            conv_pred.append(F.conv2d(single_activation, single_kernel_pred, padding='same'))
            conv_target.append(F.conv2d(single_activation, single_kernel, padding='same'))
        conv_pred_stack = torch.stack(conv_pred, dim=0).squeeze(dim=1)
        conv_target_stack = torch.stack(conv_target, dim=0).squeeze(dim=1)
        loss = F.huber_loss(conv_pred_stack, conv_target_stack) / 2
        loss /= activation.shape[0]
        if torch.cuda.is_available():
            return loss.cuda()
        return loss
