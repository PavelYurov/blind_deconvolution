# !pip install torch torchvision pillow
import math, random, os, glob
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

def make_disk_psf(ksize=21, radius=None):
    """Единообразный диск (defocus) внутри ksize×ksize."""
    if radius is None:
        radius = ksize//2 - 1
    r = radius
    ax = np.arange(ksize) - (ksize-1)/2.0
    xx, yy = np.meshgrid(ax, ax)
    mask = (xx**2 + yy**2) <= (r**2 + 1e-6)
    k = mask.astype(np.float32)
    s = k.sum()
    if s > 0: k /= s
    return k  # np.float32

def gaussian2d_at(x, y, xs, ys, sigma=0.5):
    """Небольшой гауссиан, центр (x,y), сетка (xs,ys)."""
    return np.exp(-((xs-x)**2 + (ys-y)**2)/(2*sigma*sigma))

def make_motion_psf(ksize=21, length=None, angle_deg=None, softness=0.5):
    """
    Приближение равномерной «линейной» шторки:
    кладём маленькие гауссианы вдоль линии заданной длины и угла.
    softness — ширина поперечного профиля (чем меньше, тем «жёстче» линия).
    """
    if length is None:
        length = random.randint(ksize//3, ksize-2)  # от трети до почти размера
    if angle_deg is None:
        angle_deg = random.uniform(0, 180)

    k = np.zeros((ksize, ksize), np.float32)
    cx = cy = (ksize-1)/2.0
    ax = np.arange(ksize) - cx
    xx, yy = np.meshgrid(ax, ax)

    L = max(2, length)
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    # центрируем линию вокруг центра ядра
    ts = np.linspace(-(L-1)/2, (L-1)/2, L)
    for t in ts:
        x = cx + dx * t
        y = cy + dy * t
        k += gaussian2d_at(x, y, xx+cx, yy+cy, sigma=softness)

    s = k.sum()
    if s > 0: k /= s
    return k  # np.float32

def convolve_rgb(img_np, psf_np):
    """Свёртка RGB изображения с PSF (same padding). img: HxWx3 float32[0,1]."""
    import torch
    x = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0)  # 1x3xHxW
    k = torch.from_numpy(psf_np)[None, None, :, :]            # 1x1xhxw
    C = x.shape[1]
    k = k.expand(C,1,*k.shape[-2:])
    pad = psf_np.shape[0]//2
    y = F.conv2d(x, k, padding=pad, groups=C)
    return y.squeeze(0).permute(1,2,0).clamp(0,1).numpy()

class DeblurSyntheticDataset(Dataset):
    def __init__(self, sharp_dir, crop_size=256, ksize_range=(15,31), p_noise=0.3):
        self.files = sorted(sum([glob.glob(str(Path(sharp_dir)/ext))
                                 for ext in ("*.jpg","*.jpeg","*.png","*.bmp")], []))
        self.crop = crop_size
        self.ksize_range = ksize_range
        self.p_noise = p_noise

    def __len__(self): return len(self.files)

    def _random_crop(self, img):
        w, h = img.size
        if min(w,h) < self.crop:
            s = self.crop / min(w,h)
            img = img.resize((int(w*s), int(h*s)), Image.BICUBIC)
            w, h = img.size
        i = random.randint(0, h - self.crop)
        j = random.randint(0, w - self.crop)
        return img.crop((j, i, j+self.crop, i+self.crop))

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self._random_crop(img)
        if random.random()<0.5:
            img = TF.hflip(img)
        if random.random()<0.5:
            img = TF.vflip(img)

        sharp = np.asarray(img, np.float32)/255.0

        # PSF выбор: 50/50 motion/defocus
        ksize = random.randrange(*self.ksize_range, 2)  # нечётный размер
        if random.random() < 0.5:
            psf = make_motion_psf(ksize=ksize,
                                  length=random.randint(ksize//3, ksize-2),
                                  angle_deg=random.uniform(0,180),
                                  softness=random.uniform(0.4, 1.0))
            blur_type = 0
        else:
            radius = random.uniform(ksize*0.25, ksize*0.48)
            psf = make_disk_psf(ksize=ksize, radius=radius)
            blur_type = 1

        blurred = convolve_rgb(sharp, psf)

        # немного шума (реализм)
        if random.random() < self.p_noise:
            sigma = random.uniform(0.001, 0.01)
            blurred = np.clip(blurred + np.random.normal(0, sigma, blurred.shape), 0, 1)

        sharp_t   = torch.from_numpy(sharp).permute(2,0,1).float()
        blurred_t = torch.from_numpy(blurred).permute(2,0,1).float()
        blur_type_t = torch.tensor([blur_type], dtype=torch.long)

        return blurred_t, sharp_t, blur_type_t

class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*4, base*8)

        self.u3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.u3c= DoubleConv(base*8+base*4, base*4)
        self.u2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.u2c= DoubleConv(base*4+base*2, base*2)
        self.u1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.u1c= DoubleConv(base*2+base, base)
        self.out= nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        b  = self.b(self.p3(d3))
        x  = self.u3(b);  x = self.u3c(torch.cat([x,d3], dim=1))
        x  = self.u2(x);  x = self.u2c(torch.cat([x,d2], dim=1))
        x  = self.u1(x);  x = self.u1c(torch.cat([x,d1], dim=1))
        return torch.sigmoid(self.out(x))

def psnr(x, y):
    # x,y: BxCxHxW в [0,1]
    mse = F.mse_loss(x, y, reduction='none').mean(dim=(1,2,3))
    return 10 * torch.log10(1.0 / (mse + 1e-8))

def ssim_simple(x, y, C1=0.01**2, C2=0.03**2):
    # без свёрточных окон (упрощ.) для скорости/стабильности
    mu_x = x.mean(dim=(2,3), keepdim=True)
    mu_y = y.mean(dim=(2,3), keepdim=True)
    var_x = ((x-mu_x)**2).mean(dim=(2,3), keepdim=True)
    var_y = ((y-mu_y)**2).mean(dim=(2,3), keepdim=True)
    cov_xy = ((x-mu_x)*(y-mu_y)).mean(dim=(2,3), keepdim=True)
    ssim_map = ((2*mu_x*mu_y + C1)*(2*cov_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(var_x + var_y + C2) + 1e-8)
    return ssim_map.mean(dim=(1,2,3))

class DeblurLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.5):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim = ssim_simple(pred, target).mean()
        return self.w_l1*l1 + self.w_ssim*(1.0 - ssim)
