import os
from os.path import exists

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler
import pandas as pd

from .normalizer import DataNormalizer
from .. import config as cfg


class InfiniteSampler(Sampler):
    def __init__(self, num_samples, data_source=None):
        super().__init__(data_source)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        np.random.seed(cfg.random_seed)
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed(cfg.random_seed)
                order = np.random.permutation(self.num_samples)
                i = 0


class NetCDFLoader(Dataset):
    def __init__(self, csv_files, transform=None):
        self.data_frames = [pd.read_csv(csv_file) for csv_file in csv_files]
        self.transform = transform

        # add normalization?
        # how about era data?

        #if cfg.normalization:
        #    self.data_normalizer = DataNormalizer(self.input, cfg.normalization)

    def __len__(self):
        return self.data_frames[0].shape[0]

    def __getitem__(self, idx):
        data_all = []
        for data_frame in self.data_frames:
            img_path = data_frame.iloc[idx, 0]
            variable = data_frame.iloc[idx, 1]

            # Open the NetCDF file using xarray
            data = torch.tensor(xr.open_dataset(img_path)[variable].values)
            data_all.append(data)
        
        data_all = torch.stack(data_all)
        target = data_frame.iloc[idx, 2]

        return data_all, target
