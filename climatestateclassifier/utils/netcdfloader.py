import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler

from volai.ensai.utils.normalizer import DataNormalizer
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


def nc_loadchecker(filename, data_type, image_size, keep_dss=False):
    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print('File {} not found.'.format(filename))

    try:
        # We use load_dataset instead of open_dataset because of lazy transpose
        ds = xr.open_dataset(filename, decode_times=False)
    except Exception:
        raise ValueError('Impossible to read {}.'
                         '\nPlease, check that it is a netCDF file and it is not corrupted.'.format(filename))

    ds1 = ds

    if keep_dss:
        dtype = ds[data_type].dtype
        ds = ds.drop_vars(data_type)
        ds[data_type] = np.empty(0, dtype=dtype)
        return [ds, ds1], ds1[data_type].values
    else:
        return None, ds1[data_type].values


def load_netcdf(path, data_name, data_type, data_size, keep_dss=False):
    if data_name is None:
        return None, None
    else:
        dss, data = nc_loadchecker(path, data_type, data_size,
                                   keep_dss=keep_dss)
        length = len(data[0])

        if keep_dss:
            return dss, data, length
        else:
            return data, length


class ONINetCDFLoader(Dataset):
    def __init__(self, data_root, data_in_names, data_in_types, data_in_sizes, data_out_names, data_out_types,
                 data_out_sizes, ensembles, ssis, action="test"):
        super(ONINetCDFLoader, self).__init__()

        self.data_in_types = data_in_types
        self.data_out_types = data_out_types
        self.n_ensembles = len(ensembles)
        self.ssis = ssis
        self.action = action

        self.input, self.gt_oni, self.input_oni = [], [], []

        assert len(data_in_names) == len(data_in_types) == len(data_in_sizes)

        if action == "test":
            assert len(data_out_names) == len(data_out_types) == len(data_out_sizes)

        for i in range(len(data_in_names)):
            data_in = []
            for ensemble in ensembles:
                data_path = '{:s}/input/{}/{}_ens{}.nc'.format(data_root, data_in_types[i], data_in_names[i], ensemble)
                data, _ = load_netcdf(data_path, data_in_names[i], data_in_types[i], data_in_sizes[i])
                data = data[-cfg.prediction_range-cfg.time_steps:-cfg.prediction_range, :, :]
                data_in.append(data)
            self.input.append(data_in)

        for i in range(len(data_out_names)):
            gt_oni = []
            input_oni = []
            for ensemble in ensembles:
                gt_ssi_data = []
                input_ssi_data = []
                for ssi in ssis:
                    if action == "test":
                        if ssi % 1 == 0:
                            data_path = '{:s}/output/ssi{}/{}{}.nc'.format(data_root, int(ssi), data_out_names[0], ensemble)
                        else:
                            data_path = '{:s}/output/ssi{}/{}{}.nc'.format(data_root, ssi, data_out_names[0], ensemble)
                    else:
                        data_path = '{:s}/output/ssi{}/{}{}.nc'.format(data_root, 0, data_out_names[0], ensemble)
                    data, _ = load_netcdf(data_path, data_out_names[0], data_out_types[0], data_out_sizes[0])
                    input_ssi_data.append(data[-cfg.prediction_range - cfg.time_steps:-cfg.prediction_range])
                    gt_ssi_data.append(data[-cfg.prediction_range:])
                gt_oni.append(gt_ssi_data)
                input_oni.append(input_ssi_data)
            self.gt_oni.append(gt_oni)
            self.input_oni.append(input_oni)

        if cfg.normalization:
            self.img_normalizer = DataNormalizer(self.input, cfg.normalization)
        self.input_oni_mean = torch.mean(torch.from_numpy(np.nan_to_num(self.input_oni)), dim=1)

    def __getitem__(self, index):
        # determine ssi and ensemble
        ssi_index, ensemble_index = divmod(index, self.n_ensembles)

        input_data, gt_oni_data, input_oni_data = [], [], []
        for i in range(len(self.data_in_types)):
            data = torch.from_numpy(np.nan_to_num(self.input[i][ensemble_index]))
            if cfg.normalization:
                data = self.img_normalizer.normalize(data, i)
            input_data += [data]

        for i in range(len(self.data_out_types)):
            data = torch.from_numpy(np.nan_to_num(self.gt_oni[i][ensemble_index][ssi_index]))
            gt_oni_data += [data]
            data = torch.from_numpy(np.nan_to_num(self.input_oni[i][ensemble_index][ssi_index]))
            input_oni_data += [data]

        input_data = torch.stack(input_data)
        # merge var and time dimensions
        input_data = input_data.view(*input_data.shape[:0], -1, *input_data.shape[2:])
        input_oni_data = torch.squeeze(torch.stack(input_oni_data))

        if cfg.prediction_mean:
            gt_oni_data = self.gt_oni_mean[0][ssi_index]
        elif cfg.prediction_index:
            gt_oni_data = torch.stack(gt_oni_data)[:, cfg.prediction_index, :, :]
        else:
            gt_oni_data = torch.stack(gt_oni_data)

        if self.action == "random":
            return torch.randn(input_data.shape), gt_oni_data.squeeze(), torch.randn(input_oni_data.shape), torch.tensor([self.ssis[ssi_index]])
        else:
            if cfg.add_noise:
                return input_data + (0.1 ** 0.5) * torch.randn(input_data.shape) / 100, gt_oni_data.squeeze(),\
                       input_oni_data + (0.1 ** 0.5) * torch.randn(input_oni_data.shape) / 100,\
                       torch.tensor([self.ssis[ssi_index]])
            else:
                return input_data, gt_oni_data.squeeze(), input_oni_data, torch.tensor([self.ssis[ssi_index]])

    def __len__(self):
        return self.n_ensembles * len(self.ssis)


class LocationNetCDFLoader(Dataset):
    def __init__(self, data_root, data_in_names, data_in_types, data_in_sizes, ensembles, ssis, locations, norm_to_ssi):
        super(LocationNetCDFLoader, self).__init__()

        self.data_in_names = data_in_names
        self.locations = locations
        self.ssis = ssis
        self.n_ensembles = len(ensembles)
        self.input, self.input_labels = [], []
        self.input_ssis = []
        self.input_ensembles = []

        self.xr_dss = None

        assert len(data_in_names) == len(data_in_types) == len(data_in_sizes)

        if cfg.experiment == 'historical':
            for i in range(len(data_in_names)):
                data_in = []
                for ensemble in ensembles:
                    for y in range(len(cfg.eval_years)):
                        ssi = 7.5
                        data_path = '{:s}/dghistge{}_echam6_BOT_mm_{}_{}-{}.nc'.format(data_root,
                                                                                       ensemble,
                                                                                       data_in_names[i],
                                                                                       int(cfg.eval_years[y]) - 1,
                                                                                       int(cfg.eval_years[y]) + 2)

                        if self.xr_dss is not None:
                            data, _ = load_netcdf(data_path, data_in_names[i], data_in_types[i],
                                                  data_in_sizes[i])
                        else:
                            self.xr_dss, data, _ = load_netcdf(data_path, data_in_names[i],
                                                               data_in_types[i], data_in_sizes[i],
                                                               keep_dss=True)

                        if norm_to_ssi and ssi != 0.0:
                            data = data * (norm_to_ssi / ssi)
                        if cfg.mean_input:
                            data = np.expand_dims(np.mean(data, axis=0), axis=0)
                        data_in.append(data)
                        if i == 0:
                            input_class = np.zeros(len(locations))
                            input_class[locations.index(cfg.gt_locations[y])] = 1
                            self.input_labels.append(input_class)
                            self.input_ssis.append(ssi)
                            self.input_ensembles.append(ensemble)
                self.input.append(data_in)
        elif cfg.experiment:
            for i in range(len(data_in_names)):
                data_in = []
                for y in range(len(cfg.eval_years)):

                    ssi = 7.5
                    data_path = '{:s}/{}.nc'.format(data_root, cfg.eval_years[y])

                    if self.xr_dss is not None:
                        data, _ = load_netcdf(data_path, data_in_names[i], data_in_types[i],
                                              data_in_sizes[i])
                    else:
                        self.xr_dss, data, _ = load_netcdf(data_path, data_in_names[i],
                                                           data_in_types[i], data_in_sizes[i],
                                                           keep_dss=True)

                    if norm_to_ssi and ssi != 0.0:
                        data = data * (norm_to_ssi / ssi)
                    if cfg.mean_input:
                        data = np.expand_dims(np.mean(data, axis=0), axis=0)
                    data_in.append(data)
                    if i == 0:
                        input_class = np.zeros(len(locations))
                        input_class[locations.index(cfg.gt_locations[y])] = 1
                        self.input_labels.append(input_class)
                        self.input_ssis.append(ssi)
                        self.input_ensembles.append(int(cfg.eval_years[y]))
                self.input.append(data_in)
        else:
            for i in range(len(data_in_names)):
                data_in = []
                for j in range(len(self.locations)):
                    for ssi in ssis:
                        for ensemble in ensembles:
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
                                    data_path = '{:s}/deva{}ssi{}{}_echam6_BOT_mm_{}_{}.nc'.format(data_root, converted_ssi, locations[j], ensemble, data_in_names[i], year)
                                else:
                                    data_path = '{:s}/deva{}ssi{}_echam6_BOT_mm_{}_{}.nc'.format(data_root, converted_ssi, ensemble, data_in_names[i], year)

                                if (ssi != 0.0 and locations[j] != "ne") or (locations[j] == "ne" and ssi == 0.0):
                                    if self.xr_dss is not None:
                                        data, _ = load_netcdf(data_path, data_in_names[i], data_in_types[i], data_in_sizes[i])
                                    else:
                                        self.xr_dss, data, _ = load_netcdf(data_path, data_in_names[i], data_in_types[i], data_in_sizes[i], keep_dss=True)

                                    if norm_to_ssi and ssi != 0.0:
                                        data = data * (norm_to_ssi / ssi)
                                    if cfg.mean_input:
                                        data = np.expand_dims(np.mean(data, axis=0), axis=0)
                                    data_in.append(data)
                                    if i == 0:
                                        input_class = np.zeros(len(locations))
                                        input_class[j] = 1
                                        self.input_labels.append(input_class)
                                        self.input_ssis.append(ssi)
                                        self.input_ensembles.append(ensemble)

                self.input.append(data_in)

        self.length = len(self.input_labels)

        if cfg.normalization:
            self.img_normalizer = DataNormalizer(self.input, cfg.normalization)

    def __getitem__(self, index, raw_input=False):
        input_data, input_labels = [], []

        for i in range(len(self.data_in_names)):
            data = torch.from_numpy(np.nan_to_num(self.input[i][index]))
            if cfg.normalization and not raw_input:
                data = self.img_normalizer.normalize(data, i)
            input_data += data

        input_data = torch.stack(input_data)

        if len(input_data.shape) > 3:
            input_data = input_data[0]

        input_labels = torch.from_numpy(np.nan_to_num(self.input_labels[index])).to(torch.float32)

        return input_data, input_labels, torch.tensor(self.input_ssis[index]), torch.tensor(self.input_ensembles[index])

    def __len__(self):
        return self.length #self.n_ensembles * len(self.ssis) * len(self.locations)
