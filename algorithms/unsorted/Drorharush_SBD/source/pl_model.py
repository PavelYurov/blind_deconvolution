import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from model_tools import ActivationLoss
from SBD import save_data
from dataset import QPIDataSet
from matplotlib import pyplot as plt
import numpy as np
from model_tools import Up, Down

loss_func = ActivationLoss(0.01)


def plot_example(measurement, activation, pred_activation, idx):
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
    fig.suptitle('Prediction')
    ax[0].set_title('Input', fontsize=12)
    ax[0].imshow(measurement[idx][0].cpu(), cmap='hot')

    ax[1].set_title('Network output', fontsize=12)
    ax[1].imshow(pred_activation[idx][0].cpu(), cmap='hot')

    ax[2].set_title('Target', fontsize=12)
    ax[2].imshow(activation[idx].cpu(), cmap='hot')

    for i in range(3):
        ax[i].set_axis_off()
    plt.savefig(f'C:/Users/physicsuser/Documents/SBD/pl_predictions/Pred{idx}')


class QPIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.train_dims = None

    def prepare_data(self):
        # called only on 1 GPU
        measurement_shape = (1, 128, 128)
        kernel_shape = (16, 16)

        save_data(20000, measurement_shape, kernel_shape, training=True)
        save_data(1000, measurement_shape, kernel_shape, validation=True)
        save_data(500, measurement_shape, kernel_shape, testing=True)

    def setup(self, stage=None):
        # called on every GPU
        self.train = QPIDataSet(f'{os.getcwd()}/training_dataset')
        self.val = QPIDataSet(f'{os.getcwd()}/validation_dataset')
        self.test = QPIDataSet(f'{os.getcwd()}/testing_dataset')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0)


class ActivationSys(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, input_channels=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        pad = 2
        ker_same = (5, 5)  # These values make sure the image dimensions stay the same
        self.down1 = Down(2 ** 2, 2 ** 4)
        self.down2 = Down(2 ** 4, 2 ** 6)
        self.up1 = Up(2 ** 6, 2 ** 4)
        self.up2 = Up(2 ** 4, 2 ** 2)
        self.features = nn.Sequential(

            # Input layer
            nn.Conv2d(input_channels, 2 ** 2, kernel_size=ker_same, padding=pad, bias=False),
            nn.BatchNorm2d(2 ** 2),
            nn.LeakyReLU(),

            # Hidden layers:

            self.down1,

            self.down2,

            self.up1,

            self.up2,

            # Output layer

            nn.Conv2d(2 ** 2, 1, kernel_size=ker_same, padding=pad, bias=False),
            nn.ReLU()
        )

    def forward(self, x_in):
        x = x_in / torch.linalg.norm(x_in)
        x = self.features(x)
        x = x / torch.sum(x) if torch.sum(x) != 0 else x
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        measurement, kernel, activation = batch
        pred_activation = self(measurement)
        loss = loss_func(pred_activation, activation, kernel, measurement)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        measurement, kernel, activation = batch
        pred_activation = self(measurement)
        loss = loss_func(pred_activation, activation, kernel, measurement)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        measurement, kernel, activation = batch
        pred_activation = self(measurement)
        loss = loss_func(pred_activation, activation, kernel, measurement)
        self.log("valid_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        measurement, kernel, activation = batch
        pred_activation = self(measurement)
        if batch_idx > 0:
            return pred_activation
        idxs = np.random.choice(measurement.shape[0], 10, replace=False)
        for idx in idxs:
            plot_example(measurement, activation, pred_activation, idx)
        return pred_activation

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
