import numpy as np
import torch
from torchvision import transforms
from .. import config as cfg


def data_normalization(data):
    data_std, data_mean, data_tf = [], [], []
    for i in range(len(data)):
        data_mean.append(np.nanmean(data[i]))
        data_std.append(np.nanstd(data[i]))
        data_tf.append(transforms.Normalize(mean=[data_mean[-1]], std=[data_std[-1]]))
    return data_mean, data_std, data_tf


def renormalize(data, data_mean, data_std):

    return data_std * data + data_mean
