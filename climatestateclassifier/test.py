from climatestateclassifier.utils.netcdfloader import nc_loadchecker
import numpy as np

def import_forcing(file_name, data_type):
    dss, data, lengths, sizes = zip(*[nc_loadchecker(file_name, data_type) for i in range(1)])
    data = np.nanmean(data[0], axis=(1,2))
    data = np.mean(data.reshape(-1, 3), axis=1)
    data = data / np.max(data)
    return data

