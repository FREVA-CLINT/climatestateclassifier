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

def load_netdf(file_path, variable, ts=None):
    if ts is None:
        return xr.open_dataset(file_path)[variable][:ts].values
    else:
        return xr.open_dataset(file_path)[variable].values

class NetCDFLoader(Dataset):
    def __init__(self, csv_files, data_types, timesteps, target_name):
        self.data_frames = [pd.read_csv(csv_file) for csv_file in csv_files]
        self.data_types = data_types
        self.timesteps = timesteps
        self.target_name = target_name
        
        ds = torch.tensor(load_netdf(self.data_frames[0].iloc[0,0], data_types[0]))
        self.img_sizes = [ds.shape[-2:]]
        # add normalization?
        # how about era data?

        #if cfg.normalization:
        #    self.data_normalizer = DataNormalizer(self.input, cfg.normalization)

    def __len__(self):
        return self.data_frames[0].shape[0]

    def __getitem__(self, index):
        data_all = []
        img_path = self.data_frames[0].iloc[index, 0]
        
        for idx, data_frame in enumerate(self.data_frames):
            variable = self.data_types[idx]

            # Open the NetCDF file using xarray
            data = torch.tensor(load_netdf(img_path, variable, self.timesteps[idx]))
            if data.dim() < 3:
                data = data.unsqueeze(dim=0)
            data_all.append(data)
        
        data_all = torch.concat(data_all, dim=0)
        target = data_frame[self.target_name].iloc[idx]

        return data_all, target
