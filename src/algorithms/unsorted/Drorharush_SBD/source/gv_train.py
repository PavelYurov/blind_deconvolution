# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from SBD import kernel_factory, Y_factory


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


# %%

class SBDSyntheticDataset(Dataset):
    def __init__(self, kernel_size, img_size, density, snr, size=1000):
        self.kernel_size = (kernel_size, kernel_size)
        self.img_size = (img_size, img_size)
        self.density = density
        self.snr = snr
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        Y, K, X = Y_factory(1,self.img_size, self.kernel_size, self.density, self.snr)
        X = X.toarray().astype(np.double)
        Y = Y.astype(np.double)
        K = K.astype(np.double)
        return Y, K, X


# %%
d = SBDSyntheticDataset(4, 32, 0.01, 0.99999999)



class DNNModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder =  mobilenet.MobileNetV2()
        self.decoder = decoder.DecoderSPP()
        self.flag = False
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        Y, K, X = train_batch
        # print(Y.type(), K.type(), X.type())
        z = self.encoder(Y)    
        X_pred = self.decoder(z)
        X_pred = torch.squeeze(X_pred)
        print(X_pred.shape, X_pred.sum(dim=[1,2], keepdim=True).shape)
        X_pred = X_pred/X_pred.sum(dim=[1,2], keepdim=True)
        X = X / X.sum(dim=[1,2], keepdim=True)
        # print(X_pred.shape, X.shape)
        loss = 100*F.mse_loss(X_pred, X)
        self.log('train_loss', loss)
        # print(loss)
        self.flag = True
        return loss

    def validation_step(self, val_batch, batch_idx):
        Y, K, X = val_batch
        z = self.encoder(Y)
        X_pred = self.decoder(z)
        loss = F.mse_loss(X_pred, X)
        self.log('val_loss', loss)
        if self.flag:
            y0 = Y[0].cpu().detach().numpy().squeeze()
            x0 = X[0].cpu().detach().numpy().squeeze()
            x0_pred = X_pred[0].cpu().detach().numpy().squeeze()
            print(x0.max(), x0.min(), x0_pred.max(), x0_pred.min())
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.matshow(y0)
            ax2.matshow(x0)
            ax3.matshow(x0_pred)
            plt.show()
            self.flag = False



train_loader = DataLoader(SBDSyntheticDataset(4, 32, 0.01, 0.99999999, size=10000), batch_size=100, shuffle=True)
val_loader = DataLoader(SBDSyntheticDataset(4, 32, 0.01, 0.99999999, size=40), batch_size=100, shuffle=False)

# model
model = DNNModel().double()

# model.training_step(next(iter(train_loader)),1)

# # training
trainer = pl.Trainer(gpus=1, num_nodes=1, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
    


