import os
from os.path import exists

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler

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


def nc_loadchecker(filename, data_type):
    basename = filename.split("/")[-1]

    try:
        # We use load_dataset instead of open_dataset because of lazy transpose
        ds = xr.load_dataset(filename, decode_times=True)
    except Exception:
        raise ValueError('Impossible to read {}.'
                         '\nPlease, check that it is a netCDF file and it is not corrupted.'.format(basename))

    ds1 = ds
    ds = ds.drop_vars(data_type)

    #if cfg.lazy_load:
    #     data = ds1[data_type]
    #else:
    data = ds1[data_type].values

    dims = ds1[data_type].dims
    coords = {key: ds1[data_type].coords[key] for key in ds1[data_type].coords if key != "time"}
    ds1 = ds1.drop_vars(ds1.keys())
    ds1 = ds1.drop_dims("time")

    return [ds, ds1, dims, coords], data, data.shape[0], data.shape[1:]


def load_netcdf(data_paths, data_types, keep_dss=False):
    if data_paths is None:
        return None, None
    else:
        ndata = len(data_paths)
        assert ndata == len(data_types)
        dss, data, lengths, sizes = zip(*[nc_loadchecker(data_paths[i], data_types[i]) for i in range(ndata)])

        if keep_dss:
            return dss[0], data, lengths[0], sizes
        else:
            return data, lengths[0], sizes


class NetCDFLoader(Dataset):
    def __init__(self, data_root_dirs, data_types, sample_names, sample_categories, labels):
        super(NetCDFLoader, self).__init__()

        self.labels = labels
        self.categories = sample_categories
        self.n_samples = len(sample_names)
        self.data_types = data_types
        self.input, self.input_labels, self.sample_categories, self.sample_names = [], [], [], []

        self.xr_dss = None

        for i in range(len(data_types)):
            input_data = []
            for name in sample_names:
                for category in sample_categories:
                    for j in range(len(labels)):
                        for directory in data_root_dirs:
                            # data names must be in the format: <category_name><sample_name><data_type><class_label>
                            data_path = '{:s}/{}{}{}{}.nc'.format(directory, category, name, data_types[i], labels[j])
                            if exists(data_path):
                                if self.xr_dss is not None:
                                    data, _, self.img_sizes = load_netcdf([data_path], [data_types[i]])
                                else:
                                    self.xr_dss, data, _, self.img_sizes = load_netcdf([data_path], [data_types[i]],
                                                                                       keep_dss=True)
                                input_data.append(data)
                                if i == 0:
                                    input_class = np.zeros(len(labels))
                                    input_class[j] = 1
                                    self.input_labels.append(input_class)
                                    self.sample_categories.append(category)
                                    self.sample_names.append(name)
            if input_data:
                self.input.append(input_data)
        self.length = len(self.input_labels)

        if cfg.normalization:
            self.data_normalizer = DataNormalizer(self.input, cfg.normalization)

    def __getitem__(self, index):
        input_data, input_labels = [], []

        for i in range(len(self.data_types)):
            data = torch.from_numpy(np.nan_to_num(self.input[i][index]))
            if cfg.normalization:
                data = self.data_normalizer.normalize(data, i)
            if cfg.mean_input:
                data = torch.unsqueeze(torch.mean(data, dim=1), dim=1)
            input_data += data

        input_data = torch.cat(input_data)
        input_labels = torch.from_numpy(np.nan_to_num(self.input_labels[index])).to(torch.float32)
        return input_data, input_labels, self.sample_categories[index], self.sample_names[index]

    def __len__(self):
        return self.length
