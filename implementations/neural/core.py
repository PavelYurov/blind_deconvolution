# !pip install torch torchvision pillow numpy
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.c(x)

class DIPDecoder(nn.Module):
    def __init__(self, z_channels=32, out_channels=3, nf=64, num_ups=4):
        super().__init__()
        self.head = nn.Conv2d(z_channels, nf, 3, padding=1)
        ups, n = [], nf
        for _ in range(num_ups):
            ups += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ConvBlock(n, n)]
        self.ups = nn.Sequential(*ups)
        self.tail = nn.Sequential(
            ConvBlock(n, n),
            nn.Conv2d(n, out_channels, 1)
        )
    def forward(self, z):
        x = self.head(z)
        x = self.ups(x)
        x = self.tail(x)
        return torch.sigmoid(x)  # [0,1]

def to_tensor(img_pil):
    x = np.asarray(img_pil.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)  # 1xCxHxW
    return x

def to_pil(img_t):
    x = img_t.detach().clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
    return Image.fromarray((x*255).astype(np.uint8))

def normalize_kernel(k):
    # неотрицательность + сумма=1
    k = F.relu(k)
    k = k / (k.sum(dim=(-2,-1), keepdim=True) + 1e-8)
    return k

def conv_img_with_kernel(x, k):
    # depthwise conv по каналам одной и той же PSF
    B,C,H,W = x.shape
    ks = k.shape[-1]
    pad = ks//2
    k_rep = k.expand(C,1,ks,ks)
    return F.conv2d(x, k_rep, bias=None, stride=1, padding=pad, groups=C)

def tv_loss(x):
    # изотропный TV
    dx = x[...,1:,:] - x[...,:-1,:]
    dy = x[...,:,1:] - x[...,:, :-1]
    return (dx.abs().mean() + dy.abs().mean())
