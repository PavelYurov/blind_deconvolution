import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import io


class QPIDataSet(Dataset):
    def __init__(self, dataset_path):

        self.path2dataset = dataset_path
        self.files_in_folder = os.listdir(dataset_path)  # Listing all the files in the path
        self.length = len(self.files_in_folder) // 3
        self.files_in_folder.sort()

        # The first third are the activation maps
        self.activation_map = self.files_in_folder[:self.length]
        # The 2nd third are the kernels
        self.kernel = self.files_in_folder[self.length:2 * self.length]
        # The last third are the measurements
        self.measurement = self.files_in_folder[2 * self.length:3 * self.length]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        measurement = np.load(f'{self.path2dataset}/{self.measurement[idx]}')
        kernel = np.load(f'{self.path2dataset}/{self.kernel[idx]}')
        activation = io.mmread(
            f'{self.path2dataset}/{self.activation_map[idx]}'
        ).tolil()


        measurement = torch.FloatTensor(measurement)
        kernel = torch.FloatTensor(kernel)
        activation = torch.FloatTensor(activation.A)

        if torch.cuda.is_available():
            return measurement.cuda(), kernel.cuda(), activation.cuda()
        return measurement, kernel, activation
