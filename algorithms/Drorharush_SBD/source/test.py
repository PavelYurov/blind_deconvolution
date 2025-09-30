import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
import pytorch_lightning as pl
from seg_model import mobilenet
from seg_model import decoder
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from SBD import kernel_factory, Y_factory

class SBDSyntheticDataset(Dataset):
    def __init__(self, kernel_size, img_size, density, snr):
        self.kernel_size = (kernel_size, kernel_size)
        self.img_size = (img_size, img_size)
        self.density = density
        self.snr = snr

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        Y, K, X = Y_factory(1,self.img_size, self.kernel_size, self.density, self.snr)
        X = X.toarray().astype(np.double)
        Y = Y.astype(np.double)
        K = K.astype(np.double)
        return Y, K, X


# %%
d = SBDSyntheticDataset(16, 64, 0.01, 0.99999)

Y, K, X = d[0]
print(Y.max(), Y.min())

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.matshow(Y.squeeze())
ax2.matshow(K.squeeze())
ax3.matshow(X.squeeze())
plt.show()
