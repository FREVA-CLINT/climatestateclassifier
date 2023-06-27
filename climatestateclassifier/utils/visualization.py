import numpy as np
import torch
import matplotlib.pyplot as plt


def project(data, output_range=(0, 1)):
    absmax = np.abs(data).max(axis=tuple(range(1, len(data.shape))), keepdims=True)
    data /= absmax + (absmax == 0).astype(float)
    data = (data + 1) / 2.  # range [0, 1]
    data = output_range[0] + data * (output_range[1] - output_range[0])  # range [x, y]
    return data


def heatmap(data, cmap_name="seismic"):
    cmap = plt.cm.get_cmap(cmap_name)

    if data.shape[1] in [1, 3]:
        data = data.permute(0, 2, 3, 1).detach().cpu().numpy()
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    shape = data.shape
    tmp = data.sum(axis=-1)  # Reduce channel axis

    tmp = project(tmp, output_range=(0, 255)).astype(int)
    tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[-1] = 3
    return tmp.reshape(shape).astype(np.float32)


def grid(data, nrow=3):
    bs, h, w, c = data.shape

    # Reshape to grid
    rows = bs // nrow + int(bs % nrow != 0)

    # Border around images
    data = data.reshape(rows, nrow, h, w, c)
    data = np.transpose(data, (0, 2, 1, 3, 4))
    data = data.reshape(rows * h, nrow * w, c)

    return data


def heatmap_grid(a, nrow=1, cmap_name="seismic", heatmap_fn=heatmap):
    # Compute colors
    a = heatmap_fn(a, cmap_name=cmap_name)
    return grid(a, nrow)
