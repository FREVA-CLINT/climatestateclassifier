import sys
import numpy as np
import torch

from .netcdfloader import load_netcdf
from .. import config as cfg


def prob_to_oni(output):
    top, indices = torch.topk(output, 1, dim=1)
    indices = cfg.oni_range[0] + torch.mul(indices, float(((cfg.oni_range[1] - cfg.oni_range[0]) / cfg.oni_resolution)))
    return indices.squeeze(1)


def get_references():
    # get reference/ground truth data
    references = []
    for ssi in cfg.reference_ssis:
        reference = []
        for ensemble in cfg.val_ensembles:
            if ssi % 1 == 0:
                data_path = '{:s}/output/ssi{}/{}{}.nc'.format(cfg.data_root_dir, int(ssi), cfg.out_names[0], ensemble)
            else:
                data_path = '{:s}/output/ssi{}/{}{}.nc'.format(cfg.data_root_dir, ssi, cfg.out_names[0], ensemble)
            data, _ = load_netcdf(data_path, cfg.out_names[0], cfg.out_types[0], cfg.out_sizes[0])
            reference.append(data[-cfg.prediction_range:])
        references.append(reference)
    references = torch.from_numpy(np.nan_to_num(references)).to(torch.device('cpu')).squeeze(4).squeeze(3)

    # monthly mean references
    if cfg.mm:
        references = calculate_mm_data(references)
    return references


def calculate_mm_data(data):
    kernel = np.ones(cfg.mm) / cfg.mm
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = torch.from_numpy(np.convolve(data[i, j], kernel, mode='same'))
    return data


def create_prediction_from_ckpt(model, dataset):
    partitions = get_partitions(model.parameters(), dataset.__len__())

    if partitions != 1:
        print("The data will be split in {} partitions...".format(partitions))

    input, gt_oni, input_oni, ssi, output, heatmaps = [], [], [], [], [], []

    n_elements = dataset.__len__() // partitions
    for split in range(partitions):
        data_part = []
        i_start = split * n_elements
        if split == partitions - 1:
            i_end = dataset.__len__()
        else:
            i_end = i_start + n_elements
        for i in range(4):
            data_part.append(torch.stack([dataset[j][i] for j in range(i_start, i_end)]))

        # Tensors in data_part: input, gt_oni, input_oni, ssi

        # get results from trained network
        with torch.no_grad():
            output_part = model(data_part[0].to(cfg.device), data_part[2].to(cfg.device), data_part[3].to(cfg.device))
            # create heatmaps
            if cfg.plot_heatmaps:
                heatmap = calculate_heatmap(model, output_part, data_part[0].to(cfg.device), data_part[2].to(cfg.device), data_part[3].to(cfg.device), occ_size=10, occ_stride=5,
                                            occ_pixel=-1)
                heatmap = heatmap.to(torch.device('cpu'))
                heatmaps.append(heatmap)

        output_part = output_part.to(torch.device('cpu'))
        output.append(output_part)
        data_part[0] = data_part[0].to(torch.device('cpu'))
        input.append(data_part[0])
    output = torch.cat(output)
    heatmaps = torch.cat(heatmaps)
    input = torch.cat(input)

    if cfg.loss_criterion != 'ce':
        output = output.squeeze(1)

    return output, heatmaps, input


def get_partitions(parameters, length):
    if cfg.maxmem is None:
        partitions = cfg.partitions
    else:
        model_size = 0
        for parameter in parameters:
            model_size += sys.getsizeof(parameter.storage())
        model_size = model_size * length / 1e6
        partitions = int(np.ceil(model_size * 5 / cfg.maxmem))

    if partitions > length:
        partitions = length

    return partitions


def get_enso_probabilities(input):
    enso_plus_list = []
    enso_minus_list = []
    for i in range(input.shape[0]):
        enso_plus = 0.0
        enso_minus = 0.0
        for j in range(input.shape[1]):
            august, july, june = cfg.reverse_jja_indices
            oni_mean = np.mean([input[i][j][-june], input[i][j][-july], input[i][j][-august]])
            if oni_mean >= 0.5:
                enso_plus += 1
            if oni_mean <= -0.5:
                enso_minus += 1
        enso_plus_list.append("{}%".format(int(100 * (enso_plus / input.shape[1]))))
        enso_minus_list.append("{}%".format(int(100 * (enso_minus / input.shape[1]))))
    return enso_plus_list, enso_minus_list


def calculate_heatmap(model, prediction, input_img, input_oni, inputs_ssi, occ_size=1, occ_stride=1, occ_pixel=0):
    # get the width and height of the image
    channels, width, height = input_img.shape[-3], input_img.shape[-2], input_img.shape[-1]
    # setting the output image width and height
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))

    # create a white image of sizes we defined
    heatmap = torch.zeros((input_img.shape[0], input_img.shape[1], output_height, output_width))

    # iterate all the pixels in each column
    for b in range(input_img.shape[0]):
        for c in range(input_img.shape[1]):
            for h in range(height):
                for w in range(width):

                    h_start = h * occ_stride
                    w_start = w * occ_stride
                    h_end = min(height, h_start + occ_size)
                    w_end = min(width, w_start + occ_size)

                    if (w_end) >= width or (h_end) >= height:
                        continue

                    input_image = input_img[b, :, :, :].clone().detach()

                    # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
                    input_image[c, w_start:w_end, h_start:h_end] = occ_pixel
                    input_image = input_image.unsqueeze(0)

                    # run inference on modified image
                    output = model(input_image, input_oni, inputs_ssi)
                    prob = 1 - torch.mean((output - prediction) ** 2)

                    # setting the heatmap location to probability value
                    heatmap[b, c, h, w] = prob
    return heatmap
