from climatestateclassifier.utils.netcdfloader import nc_loadchecker
import numpy as np

def import_forcing(file_name, data_type):
    dss, data, lengths, sizes = zip(*[nc_loadchecker(file_name, data_type) for i in range(1)])
    return data[0]

