import math

import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap

from .. import config as cfg
import csv

from ..test import import_forcing


def save_results_as_csv(eval_dir, eval_name, labels, outputs, categories, sample_names):
    with open('{}/overview/{}.csv'.format(eval_dir, eval_name), 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["Label", "Output", "Category", "Name"])

        for i in range(labels.shape[0]):
            writer.writerow([labels[i].item(), outputs[i].item(), categories[i], sample_names[i]])


def read_results_from_csv(eval_dir, eval_name):
    labels, outputs, categories, sample_names = [], [], [], []
    with open('{}/overview/{}.csv'.format(eval_dir, eval_name), 'r') as file:
        csvreader = csv.reader(file)
        i = 0
        for row in csvreader:
            if i != 0:
                labels.append(torch.tensor([int(row[0])]))
                outputs.append(torch.tensor([int(row[1])]))
                categories.append(row[2])
                sample_names.append(row[3])
            i = i+1
    return torch.cat(labels), torch.cat(outputs), categories, sample_names


def plot_prediction_overview(outputs, labels, eval_name):
    # Clean predictions
    total_numbers = []
    total_numbers_predictions = []
    correct_predictions = []
    false_predictions = []
    metric = torchmetrics.classification.BinaryStatScores()
    for i in range(len(cfg.labels)):
        total_numbers.append(torch.sum(labels[:, i]).int().item())
        total_numbers_predictions.append(torch.sum(cleaned_predictions[:, i]).int().item())
        correct_predictions.append(metric(cleaned_predictions[:, i], labels[:, i])[0].item())
        false_predictions.append(metric(cleaned_predictions[:, i], labels[:, i])[1].item())

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    column_names = [label_name for label_name in cfg.label_names]
    column_names.append("$\\bf{Total}$")
    total_numbers.append(sum(total_numbers))
    total_numbers_predictions.append(sum(total_numbers_predictions))
    correct_predictions.append(sum(correct_predictions))
    false_predictions.append(sum(false_predictions))

    d = {
        '$\\bf{Labels}$': column_names,
        '$\\bf{Total Ground Truth}$': total_numbers,
        '$\\bf{Total Predicted}$': total_numbers_predictions,
        '$\\bf{Correct Classifications}$': correct_predictions,
        '$\\bf{False Classifications}$': false_predictions
    }
    # set colors
    colors = []
    for i in range(len(column_names)):
        if i % 2 == 0:
            colors.append(len(d.keys()) * ["#caccce"])
        else:
            colors.append(len(d.keys()) * ["#e6e7e9"])

    df = pd.DataFrame(data=d)
    table = ax.table(colWidths=len(d.keys()) * [0.2], cellText=df.values, colLabels=df.columns, loc='center',
                     colColours=len(d.keys()) * ["#002f4a"], cellColours=colors)
    table.set_fontsize(30)
    table.scale(1.5, 1.5)
    for i in range(len(d.keys())):
        table[(0, i)].get_text().set_color('white')
    fig.tight_layout()
    fig.suptitle("{} - Total Accuracy: {} %".format(eval_name,
                                                    round(100 * (correct_predictions[-1] / total_numbers[-1]), 2)))
    plt.savefig("{}/overview/{}.pdf".format(cfg.eval_dir, eval_name), bbox_inches='tight')
    plt.clf()


def plot_single_predictions(outputs, labels, categories, sample_names, eval_name):
    # Clean predictions

    prediction_labels = []
    gt_labels = []
    prediction_colors = []
    for i in range(outputs.shape[0]):
        gt_labels.append(cfg.label_names[labels[i]])
        prediction_labels.append(cfg.label_names[outputs[i]])
        if gt_labels[-1] == prediction_labels[-1]:
            prediction_colors.append(['white', 'white', 'white', 'lightgreen'])
        else:
            prediction_colors.append(['white', 'white', 'white', 'red'])

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    d = {
        '$\\bf{Samples}$': sample_names,
        '$\\bf{Categories}$': categories,
        '$\\bf{Ground Truth}$': gt_labels,
        '$\\bf{Prediction}$': prediction_labels,
    }

    df = pd.DataFrame(data=d)
    table = ax.table(colWidths=len(d.keys()) * [0.2], cellText=df.values, colLabels=df.columns, loc='center',
                     colColours=len(d.keys()) * ["#002f4a"], cellColours=prediction_colors)
    table.set_fontsize(30)
    for i in range(len(d.keys())):
        table[(0, i)].get_text().set_color('white')
    fig.tight_layout()
    plt.savefig("{}/total/{}.pdf".format(cfg.eval_dir, eval_name), bbox_inches='tight')
    plt.clf()


def plot_class_predictions(outputs, labels, eval_name):
    # Clean predictions
    class_predictions = [[0 for j in range(len(cfg.labels))] for i in range(len(cfg.labels))]
    for i in range(outputs.shape[0]):
        class_predictions[labels[i]][outputs[i]] += 1

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    prediction_colors = []

    d = {' ': cfg.label_names}

    for i in range(len(cfg.labels)):
        d['{}'.format(cfg.label_names[i])] = ['{} %'.format(
            round(100 * (pred / sum(class_predictions[i])), 2)) for pred in class_predictions[i]]
        prediction_colors.append(["#002f4a"] + len(cfg.labels) * ['red'])
        prediction_colors[i][i + 1] = 'green'

    df = pd.DataFrame(data=d)
    table = ax.table(colWidths=len(d.keys()) * [0.2], cellText=df.values, colLabels=df.columns, loc='center',
                     colColours=len(d.keys()) * ["#002f4a"], cellColours=prediction_colors)
    table.set_fontsize(30)
    for i in range(len(d.keys())):
        for j in range(len(d.keys())):
            if i == 0 or j == 0:
                table[(j, i)].get_text().set_color('white')
            elif i == j:
                table[(j, i)].get_text().set_color('black')
            else:
                table[(j, i)].get_text().set_color('white')

    fig.tight_layout()
    plt.savefig("{}/overview/{}_classes.pdf".format(cfg.eval_dir, eval_name), bbox_inches='tight')
    plt.clf()


def plot_predictions_by_category(outputs, labels, categories, eval_name):

    class_predictions = [[0 for j in range(len(cfg.labels) * len(cfg.val_categories))] for i in range(len(cfg.labels))]
    for i in range(len(cfg.val_categories)):
        for k in range(labels.shape[0]):
            if cfg.val_categories[i] == categories[k]:
                class_predictions[labels[k]][outputs[k] + (i * len(cfg.labels))] += 1.0 / len(cfg.val_samples)

    for i in range(len(class_predictions)):
        for j in range(len(class_predictions[i])):
            class_predictions[i][j] = "{} %".format(
                int(math.ceil(100 * (100 * class_predictions[i][j])) / 100) if math.ceil(
                    100 * (100 * class_predictions[i][j])) / 100 % 1 == 0 else math.ceil(
                    100 * (100 * class_predictions[i][j])) / 100)

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    category_labels = []
    class_labels = []
    for i in range(len(cfg.val_categories)):
        for j in range(len(cfg.labels)):
            category_labels.append("{}".format(cfg.val_categories[i]))
            class_labels.append("{}".format(cfg.label_names[j]))
    d = {'Category': category_labels,
         'Label': class_labels}

    prediction_colors = []

    for i in range(len(cfg.labels)):
        d['{}'.format(cfg.label_names[i])] = class_predictions[i]

    for i in range(len(class_labels)):
        row_colors = 2 * ["#002f4a"]
        for j in range(len(cfg.labels)):
            if class_predictions[j][i] == "0 %":
                row_colors.append('white')
                class_predictions[j][i] = "-"
            elif class_labels[i] == cfg.label_names[j]:
                row_colors.append('green')
            else:
                row_colors.append('red')
        prediction_colors.append(row_colors)

    df = pd.DataFrame(data=d)
    table = ax.table(colWidths=len(d.keys()) * [0.2], cellText=df.values, colLabels=df.columns, loc='center',
                     colColours=2 * ['white'] + (len(d.keys()) - 2) * ["#002f4a"], cellColours=prediction_colors)
    table.set_fontsize(30)
    for i in range(len(d.keys())):
        for j in range(len(prediction_colors) + 1):
            if i == 0 or j == 0 or i == 1:
                table[(j, i)].get_text().set_color('white')
            elif i > 1 and prediction_colors[j - 1][i] == 'red':
                table[(j, i)].get_text().set_color('white')
            if j == 0 and (i == 0 or i == 1):
                table[(j, i)].get_text().set_color('black')

    fig.tight_layout()
    plt.savefig("{}/overview/{}_categories.pdf".format(cfg.eval_dir, eval_name), bbox_inches='tight')
    plt.clf()


def plot_predictions_by_category_graph(outputs, categories, eval_name):
    levels_forcing = [0.0, 0.01, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3]
    norm_forcing = matplotlib.colors.BoundaryNorm(levels_forcing, 13)
    colors_forcing = ["#FFFFFF", "#FFFBAA", "#FFF88A", "#FFF56A", "#FFF24A", "#FFEF2A", "#FFEC0A", "#FFC408", "#FF9606", "#FF6704", "#FF3802", "#800026"]

    cmap_forcing = matplotlib.colors.ListedColormap(
        colors_forcing)
    cmap_forcing = cmap_forcing(np.arange(cmap_forcing.N))
    cmap_forcing = ListedColormap(cmap_forcing)

    years = [int(cat) for cat in cfg.val_categories]

    class_predictions = {}
    for name in cfg.label_names:
        class_predictions[name] = [0 for i in range(len(cfg.val_categories))]

    for i in range(outputs.shape[0]):
        try:
            class_predictions[cfg.label_names[outputs[i]]][cfg.val_categories.index(categories[i])] += 1.0 / len(cfg.val_samples)
        except ValueError:
            pass

    global_aod = import_forcing("/home/joe/PycharmProjects/climatestateclassifier/paper/tauttl.nc", "tauttl")[12*(years[0]-1850):12*(years[0]-1850 + len(years) - (years[-1]-1999))]
    global_mean_aod = np.nanmean(global_aod, axis=(1, 2))
    global_mean_aod = np.mean(global_mean_aod.reshape(-1, 12), axis=1)
    global_aod = global_aod * 0.0# np.max(global_mean_aod)
    global_aod[0:800] += 3.0
    global_aod[800:] -= 3.0
    class_predictions["Global AOD"] = global_mean_aod
    print(global_aod.shape)

    # Calculate the width of each bar
    bar_width = 1

    # Create a figure and axis
    fig, ax = plt.subplots(nrows=2, figsize=(15, 4.5))
    fig.tight_layout()

    #ax[0].set_title("MPI-GE Member Classifications")
    #ax[1].set_title("Stratospheric Aerosol Optical Depth Field", loc="bottom")

    # Plot the first time series
    label_names = ["Southern Hemisphere", "Tropics", "Northern Hemisphere"]
    y_axes = ["SHext -", "trop -", "NHext -"]
    class_colors = ["gray", "red", "purple", "blue"]

    height = 0.1
    colors_prob = ["#FFFFFF", "#FFF0F0", "#FFE1E1", "#FFD2D2", "#FFC3C3", "#FFB4B4", "#FFA5A5", "#FF9696", "#FF8787", "#FF7878", "#FF6969", "#FF0000"]

    cmap_prob = matplotlib.cm.RdBu_r
    current_bottom = 0.0
    for name, color in zip(label_names, class_colors):
        for year, value in zip(years, class_predictions[name]):
            img = ax[0].bar(year, height, color=cmap_prob(value), width=bar_width, align='center', bottom=current_bottom)
        current_bottom += height

    levels_prob = [0.0, 0.08, 0.16, 0.25, 0.33, 0.41, 0.5, 0.58, 0.66, 0.75, 0.83, 0.91, 1.0]
    #norm_prob = matplotlib.colors.BoundaryNorm(levels_prob, 13)

    #cmap_prob = matplotlib.colors.ListedColormap(
    #    colors_prob)
    #cmap_prob = cmap_prob(np.arange(cmap_prob.N))
    #cmap_prob = ListedColormap(cmap_prob)

    sm = ScalarMappable(cmap=cmap_prob)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax[0], location="right", ticks=[0, 0.25, 0.5, 0.75, 1])

    volcanoes = {
        1883: "Krakatau",
        1902: "Santa Maria",
        1912: "Katmai",
        1963: "Agung",
        1974: "Fuego",
        1982: "El Chichon",
        1991: "Pinatubo"
    }

    # Set the x-axis limits and labels
    ax[0].set_xlim(years[0], years[-1])

    # Set the y-axis limits and label
    ax[0].set_ylim(0, len(label_names) * height)
    ax[0].yaxis.set_visible(False)

    current_bottom = height / 2
    for name in y_axes:
        ax[0].text(years[0], current_bottom, name, ha='right', va='center')
        current_bottom += height

    img = ax[1].imshow(np.flip(np.transpose(global_aod.squeeze()), axis=0), extent=[years[0], years[-1], -89, 89], interpolation='nearest', aspect='auto', cmap=cmap_prob)
    #ax[0].set_aspect(40)
    #ax[1].set_aspect(0.06)
    ax[1].yaxis.set_visible(False)
    ann_pos = [-6.1, 14.45, 58.16, -8.2, 14.28, 17.36, 15.13]

    for key, value, pos in zip(volcanoes.keys(), volcanoes.values(), ann_pos):
        ax[1].annotate(value, xy=(key, pos), arrowprops={"headwidth": 5, "headlength": 5}, ha='center', fontsize=10)

    cbar = fig.colorbar(img, ax=ax[1], location="right", ticks=[-3, -1.5, 0.0, 1.5, 3])
    y_axes = ["50°S -", "0° -", "50°N -"]

    current_bottom = -50
    for name in y_axes:
        ax[1].text(years[0], current_bottom, name, ha='right', va='center')
        current_bottom += 50

    # Show the plot
    plt.savefig("{}/overview/{}_categories_graph.pdf".format(cfg.eval_dir, eval_name), bbox_inches='tight')
    plt.clf()


def plot_predictions_by_category_timeseries(outputs, categories, eval_name):
    years = [int(cat) for cat in cfg.val_categories]

    cmip6aod = import_forcing("/home/joe/PycharmProjects/climatestateclassifier/paper/cmip6aod.nc", "aod")
    cmip6aod_mean = np.nanmean(cmip6aod, axis=(1, 2))
    #cmip6aod_mean = np.convolve(cmip6aod_mean, np.ones(12) / 12, mode='same')
    cmip6aod_mean = cmip6aod_mean[12*(years[0]-1850):12*(years[0]-1850 + len(years) - (years[-1]-1999))]

    cmip5aod = import_forcing("/home/joe/PycharmProjects/climatestateclassifier/paper/tauttl.nc", "tauttl")
    cmip5aod_mean = np.nanmean(cmip5aod, axis=(1, 2))
    # cmip6aod_mean = np.convolve(cmip6aod_mean, np.ones(12) / 12, mode='same')
    cmip5aod_mean = cmip5aod_mean[12 * (years[0] - 1850):12 * (years[0] - 1850 + len(years) - (years[-1] - 1999))]

    glossacaod = import_forcing("/home/joe/PycharmProjects/climatestateclassifier/paper/GloSSAC_V2.0.nc", "Glossac_Aerosol_Optical_Depth")
    glossacaod_mean = np.nanmean(glossacaod, axis=(1, 2))
    #glossacaod_mean = np.convolve(glossacaod_mean, np.ones(12) / 12, mode='same')
    glossacaod_mean = glossacaod_mean[:12 * 21]

    eval_predictions = {}
    for name in outputs.keys():
        class_predictions = {}
        for label in cfg.label_names:
            class_predictions[label] = [0 for i in range(len(cfg.val_categories))]

        for i in range(outputs[name].shape[0]):
            try:
                class_predictions[cfg.label_names[outputs[name][i]]][cfg.val_categories.index(categories[name][i])] += 1
            except ValueError:
                pass
        eval_predictions[name] = class_predictions

    combined_predictions = {}
    for label in cfg.label_names:
        combined_predictions[label] = [0 for i in range(len(cfg.val_categories))]
    # calculate means
    for key, value in eval_predictions.items():
        for key2, value2 in value.items():
            for i in range(len(cfg.val_categories)):
                combined_predictions[key2][i] += value2[i]

    total_numbers = [0 for i in range(len(cfg.val_categories))]
    for i in range(len(cfg.val_categories)):
        for key,value in combined_predictions.items():
            total_numbers[i] += value[i]

    fig = plt.figure(figsize=(10, 2))

    color = cfg.timeseries_colors[0]
    ax1 = fig.add_axes([0, 0, 1, 1])

    ax2 = ax1.twinx()
    ax1.set_ylabel('Predicted Eruptions in %', color=color)
    ax1.bar(years, [((combined_predictions["Northern Hemisphere"][i] + combined_predictions["Southern Hemisphere"][i] +
            combined_predictions["Tropics"][i]) / total_numbers[i]) * 100 if total_numbers[i] != 0 else 0 for i in
            range(len(combined_predictions["No Eruption"]))],
            color=cfg.timeseries_colors[0], label="Predicted Eruptions")
    ax1.tick_params(axis='y', labelcolor=color)
    color = 'black'
    ax2.set_ylabel('Average Global AOD', color=color)  # we already handled the x-label with ax1
    #ax2.plot([years[0] + (1.0/12) * x for x in range(len(cmip5aod_mean))], [aod for aod in cmip5aod_mean], "y--", label="CMIP5 AOD")
    ax2.plot([years[0] + (1.0/12) * x for x in range(len(cmip6aod_mean))], [aod * 10 for aod in cmip6aod_mean], "r--", label="CMIP6 AOD")
    ax2.plot([1979 + (1.0/12) * x for x in range(len(glossacaod_mean))], glossacaod_mean, "b--", label="GloSSAC AOD")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.margins(x=0, y=0.01)
    ax1.margins(x=0)
    plt.figlegend(loc='upper center', fancybox=True, framealpha=1, shadow=True, fontsize='small',  bbox_to_anchor=(0.5, 0.99), ncol=2)
    ax1.set_title("Evaluation {}".format(cfg.timeseries_names[0]))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Set the x-axis limits and labels
    plt.savefig("{}/{}_timeseries.pdf".format(cfg.eval_dir, eval_name), bbox_inches='tight')
    plt.clf()


def plot_single_explanation(explanations, ax, dims, pad=0.13):
    # color map
    if not cfg.cmap_colors:
        cmap = matplotlib.cm.RdBu
    else:
        cmap = matplotlib.colors.ListedColormap(cfg.cmap_colors)
    new_cmap = cmap(np.arange(cmap.N))
    new_cmap = ListedColormap(new_cmap)

    vmin = 0
    vmax = 1
    for time in range(explanations.shape[0]):
        # axes[i].axis('off')
        gl = ax[time].gridlines(crs=ccrs.Robinson(), draw_labels=False, linewidth=0.1)
        gl.top_labels = False
        gl.right_labels = False
        ax[time].add_feature(cartopy.feature.COASTLINE, edgecolor="black", linewidth=0.3)
        ax[time].add_feature(cartopy.feature.BORDERS, edgecolor="black", linestyle="--", linewidth=0.3)

        ax[time].gridlines(crs=ccrs.Robinson(), draw_labels=False, linewidth=0.1)
        plot = ax[time].pcolormesh(dims["lon"], dims["lat"], 5000 * explanations[time, :, :].detach().numpy(),
                                   vmin=vmin, vmax=vmax,
                                   transform=ccrs.PlateCarree(), shading='auto', cmap=new_cmap, linewidth=0,
                                   rasterized=True)

        cb = plt.colorbar(plot, location="bottom", ax=ax[time], fraction=0.09, pad=pad)
        cb.ax.tick_params(labelsize=5)
        cb.ax.ticklabel_format(useOffset=True, style='plain')

        cb.ax.set_title("{}".format("Relevance (unitless)"), fontsize=5)


def plot_explanations(inputs, dims, gt, outputs, sample_names, category_names, all_explanations, eval_name):
    for i in range(inputs.shape[0]):
        n_rows = cfg.time_steps if not cfg.mean_input else 1
        n_cols = len(cfg.data_types) * (len(cfg.explanation_names) + 1) + 1
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols + 1.5, n_rows * 1.25),
                               subplot_kw={"projection": ccrs.Robinson()}, squeeze=False)

        gt_index = torch.argmax(gt[i])
        gt_class = cfg.label_names[gt_index]
        pred_index = torch.argmax(outputs[i])
        pred_class = cfg.label_names[pred_index]

        # Create data frame for table
        d = {'$\\bf{Label}$': ["Ground Truth", "Prediction"]}
        pred_colors = ['white']
        if gt_class == pred_class:
            pred_colors.append("lightgreen")
        else:
            pred_colors.append("red")
        key = 'Sample {}, Cat {}'.format(sample_names[i], category_names[i])
        key = '$\\bf{' + key + '}$'
        d[key] = [gt_class, pred_class]
        colors = [2 * ['white'], pred_colors]
        df = pd.DataFrame(data=d)
        table = ax[n_rows // 2, 0].table(colWidths=[0.3, 0.5], cellText=df.values, colLabels=df.columns, loc='center',
                                         colColours=len(d.keys()) * ["#002f4a"], cellColours=colors)

        table.set_fontsize(35)
        table.scale(1.5, 1.5)
        for k in range(len(d.keys())):
            table[(0, k)].get_text().set_color('white')

        # Plot inputs
        for time in range(inputs.shape[1]):
            ax[time, 0].axis('off')
            ax[time, 0].axis('tight')
            for var in range(inputs.shape[2]):
                vmin = -3#torch.min(raw_input[i, var]) / 2
                vmax = -vmin
                if cfg.data_types[var] == 'pr':
                    cmap = "RdBu"
                else:
                    cmap = "RdBu_r"
                col = var * (len(cfg.explanation_names) + 1) + 1
                gl = ax[time, col].gridlines(crs=ccrs.Robinson(), draw_labels=False, linewidth=0.1)
                gl.top_labels = False
                gl.right_labels = False
                ax[time, col].add_feature(cartopy.feature.COASTLINE, edgecolor="black", linewidth=0.3)
                ax[time, col].add_feature(cartopy.feature.BORDERS, edgecolor="black", linestyle="--", linewidth=0.3)

                plot = ax[time, col].pcolormesh(dims["lon"], dims["lat"],
                                                inputs[i][time, var, :, :].detach().numpy(),
                                                cmap=cmap, transform=ccrs.PlateCarree(), shading='auto', vmin=vmin,
                                                vmax=vmax, linewidth=0, rasterized=True)
                cb = plt.colorbar(plot, location="bottom", ax=ax[time, col], fraction=0.09, pad=0.2)
                cb.ax.tick_params(labelsize=5)
                cb.ax.ticklabel_format(useOffset=True, style='plain')
                cb.ax.set_title("{}".format("Sea Surface Temperature Anomaly (°C)"), fontsize=5)
        for exp in range(all_explanations.shape[0]):
            for var in range(all_explanations.shape[2]):
                plot_single_explanation(all_explanations[exp, i, var],
                                        ax[:, var * (all_explanations.shape[0] + 1) + exp + 1 + 1], dims, pad=0.2)

        fig.tight_layout()
        fig.savefig(
            '{}/explanations/{}{}{}{}.jpg'.format(cfg.eval_dir, eval_name, category_names[i], sample_names[i],
                                                     gt_class.replace(' ', '')),
            dpi=800)
        plt.close(fig)
        plt.clf()
