import os

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

    if not os.path.isfile(filename):
        print('File {} not found.'.format(filename))

    try:
        # We use load_dataset instead of open_dataset because of lazy transpose
        ds = xr.load_dataset(filename, decode_times=True)

    except Exception:
        raise ValueError('Impossible to read {}.'
                         '\nPlease, check that it is a netCDF file and it is not corrupted.'.format(basename))

    ds1 = ds
    ds = ds.drop_vars(data_type)

    if cfg.lazy_load:
        data = ds1[data_type]
    else:
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
    def __init__(self, data_root, data_types, samples, ssis, labels, norm_to_ssi):
        super(NetCDFLoader, self).__init__()

        self.labels = labels
        self.ssis = ssis
        self.n_samples = len(samples)
        self.input, self.input_labels = [], []
        self.input_ssis = []
        self.input_samples = []
        self.data_types = data_types

        self.xr_dss = None

        if cfg.experiment == 'historical':
            for i in range(len(data_types)):
                data_in = []
                for sample in samples:
                    for y in range(len(cfg.eval_years)):
                        ssi = 7.5
                        data_path = '{:s}/dghistge{}_echam6_BOT_mm_{}_{}-{}.nc'.format(data_root,
                                                                                       sample,
                                                                                       data_types[i],
                                                                                       int(cfg.eval_years[y]) - 1,
                                                                                       int(cfg.eval_years[y]) + 2)

                        if self.xr_dss is not None:
                            data, _, self.img_sizes = load_netcdf([data_path], [data_types[i]])
                        else:
                            self.xr_dss, data, _, self.img_sizes = load_netcdf([data_path], [data_types[i]],
                                                                               keep_dss=True)

                        if norm_to_ssi and ssi != 0.0:
                            data = data * (norm_to_ssi / ssi)
                        if cfg.mean_input:
                            data = np.expand_dims(np.mean(data, axis=0), axis=0)
                        data_in.append(data)
                        if i == 0:
                            input_class = np.zeros(len(labels))
                            input_class[labels.index(cfg.gt_locations[y])] = 1
                            self.input_labels.append(input_class)
                            self.input_ssis.append(ssi)
                            self.input_samples.append(sample)
                self.input.append(data_in)
        elif cfg.experiment:
            for i in range(len(data_types)):
                data_in = []
                for y in range(len(cfg.eval_years)):

                    ssi = 7.5
                    data_path = '{:s}/{}.nc'.format(data_root, cfg.eval_years[y])

                    if self.xr_dss is not None:
                        data, _, self.img_sizes = load_netcdf([data_path], [data_types[i]])
                    else:
                        self.xr_dss, data, _, self.img_sizes = load_netcdf([data_path], [data_types[i]], keep_dss=True)

                    if norm_to_ssi and ssi != 0.0:
                        data = data * (norm_to_ssi / ssi)
                    if cfg.mean_input:
                        data = np.expand_dims(np.mean(data, axis=0), axis=0)
                    data_in.append(data)
                    if i == 0:
                        input_class = np.zeros(len(labels))
                        input_class[labels.index(cfg.gt_locations[y])] = 1
                        self.input_labels.append(input_class)
                        self.input_ssis.append(ssi)
                        self.input_samples.append(int(cfg.eval_years[y]))
                self.input.append(data_in)
        else:
            for i in range(len(data_types)):
                data_in = []
                for j in range(len(self.labels)):
                    for ssi in ssis:
                        for sample in samples:
                            if ssi % 1 == 0:
                                converted_ssi = int(ssi)
                            else:
                                converted_ssi = ssi

                            if ssi == 0.0:
                                years = cfg.train_years
                            else:
                                years = [cfg.train_years[1]]
                            for year in years:
                                if ssi != 0.0:
                                    data_path = '{:s}/deva{}ssi{}{}_echam6_BOT_mm_{}_{}.nc'.format(data_root, converted_ssi, labels[j], sample, data_types[i], year)
                                else:
                                    data_path = '{:s}/deva{}ssi{}_echam6_BOT_mm_{}_{}.nc'.format(data_root, converted_ssi, sample, data_types[i], year)

                                if (ssi != 0.0 and labels[j] != "ne") or (labels[j] == "ne" and ssi == 0.0):
                                    if self.xr_dss is not None:
                                        data, _, self.img_sizes = load_netcdf([data_path], [data_types[i]])
                                    else:
                                        self.xr_dss, data, _, self.img_sizes = load_netcdf([data_path], [data_types[i]],
                                                                                           keep_dss=True)

                                    if norm_to_ssi and ssi != 0.0:
                                        data = data * (norm_to_ssi / ssi)
                                    data_in.append(data)
                                    if i == 0:
                                        input_class = np.zeros(len(labels))
                                        input_class[j] = 1
                                        self.input_labels.append(input_class)
                                        self.input_ssis.append(ssi)
                                        self.input_samples.append(sample)

                self.input.append(data_in)

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
        return input_data, input_labels, torch.tensor(self.input_ssis[index]), torch.tensor(self.input_samples[index])

    def __len__(self):
        return self.length
