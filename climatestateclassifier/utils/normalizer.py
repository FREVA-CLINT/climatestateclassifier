import numpy as np
from torchvision import transforms
from .. import config as cfg


class DataNormalizer:
    def __init__(self, data, normalization, data_stats=None):
        data_std, data_mean, data_min, data_max, self.data_tf = [], [], [], [], []

        self.normalization = normalization

        for i in range(len(data)):
            if data is not None:
                data_std.append(np.nanstd(data[i]))
                data_mean.append(np.nanmean(data[i]))
            else:
                data_std.append(data_stats["std"][i])
                data_mean.append(data_stats["mean"][i])
            if normalization == 'std':
                self.data_tf.append(transforms.Normalize(mean=[data_mean[-1]], std=[data_std[-1]]))
            elif normalization == 'img':
                self.data_tf.append(transforms.Normalize(mean=0.5, std=0.5))
        self.data_stats = {"mean": data_mean, "std": data_std}

    def normalize(self, data, index):
        if self.normalization == 'std':
            return self.data_tf[index](data)
        else:
            return data

    def renormalize(self, data, index):
        if self.normalization == 'std':
            return self.data_stats["std"][index] * data + self.data_stats["mean"][index]
        else:
            return data
