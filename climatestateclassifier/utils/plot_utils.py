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
            i = i + 1
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


def format_label(label):
    if label == "Tropics":
        return "TR"
    elif label == "Northern Hemisphere":
        return "NHE"
    elif label == "Southern Hemisphere":
        return "SHE"
    else:
        return "UF"


def plot_predictions_by_category(outputs, labels, categories, eval_name):
    for i in range(len(outputs)):
        if outputs[i] == 0:
            outputs[i] = 1
        elif outputs[i] == 1:
            outputs[i] = 0
        if labels[i] == 0:
            labels[i] = 1
        elif labels[i] == 1:
            labels[i] = 0

    class_predictions = [[0 for j in range(1 + (len(cfg.labels) - 1) * (len(cfg.val_categories) - 3))] for i in
                         range(len(cfg.labels))]
    print(class_predictions.__len__())
    print(class_predictions[0].__len__())
    for i in range(len(cfg.val_categories)):
        for k in range(labels.shape[0]):
            if cfg.val_categories[i] == categories[k]:
                if categories[k].split("_")[1] == "0":
                    class_predictions[outputs[k]][0] += 1.0 / (3*len(cfg.val_samples))
                else:
                    class_predictions[outputs[k]][labels[k] + ((i - 3) * (len(cfg.labels) - 1)) + 1] += 1.0 / len(
                        cfg.val_samples)

    for i in range(len(class_predictions)):
        for j in range(len(class_predictions[i])):
            class_predictions[i][j] = "{} %".format(
                int(math.ceil(100 * (100 * class_predictions[i][j])) / 100) if math.ceil(
                    100 * (100 * class_predictions[i][j])) / 100 % 1 < 0.5 else int(math.ceil(
                    100 * (100 * class_predictions[i][j])) / 100) + 1)

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    category_labels = []
    class_labels = []
    label_names = ['Northern Hemisphere', 'Tropics', 'Southern Hemisphere', 'No Eruption']
    for i in range(len(cfg.val_categories)):
        for j in range(len(cfg.labels)):
            category_labels.append("{}".format(cfg.val_categories[i]))
            class_labels.append("{}".format(label_names[j]))
    final_labels = ["0 Tg S - UF"] + ["{}Tg S - {}".format(label.split("_")[1], format_label(class_label)) for label, class_label in
                                      zip(category_labels,
                                          class_labels) if label.split("_")[1] != "0" and class_label != "No Eruption"]

    print(final_labels)
    d = {'Ensemble': final_labels}

    prediction_colors = []

    for i in range(len(cfg.labels)):
        d['{}'.format(format_label(label_names[i]))] = class_predictions[:][i]

    avg_scores = []

    for i in range(len(final_labels)):
        row_colors = ["#002f4a"]
        for j in range(len(cfg.labels)):
            if class_predictions[j][i] == "0 %":
                row_colors.append('white')
                class_predictions[j][i] = "-"
            elif format_label(label_names[j]) in final_labels[i]:
                row_colors.append('green')
                if "UF" in final_labels[i]:
                    avg_scores += 3*[int(class_predictions[j][i].split(" ")[0])]
                else:
                    avg_scores.append(int(class_predictions[j][i].split(" ")[0]))

            else:
                row_colors.append('red')
        prediction_colors.append(row_colors)

    df = pd.DataFrame(data=d)
    table = ax.table(colWidths=[0.2] + (len(d.keys()) - 1) * [0.1], cellText=df.values, colLabels=df.columns,
                     loc='center',
                     colColours=2 * ['Gray'] + (len(d.keys()) - 2) * ["Gray"])

    table.set_fontsize(17)
    table.scale(2.5, 2.0)
    print(d.keys())
    for i in range(len(d.keys())):
        for j in range(len(prediction_colors) + 1):
            if i > 0 and prediction_colors[j - 1][i] == 'red' and j > 0:
                table[(j, i)].get_text().set_color('red')
            elif i > 0 and prediction_colors[j - 1][i] == 'green' and j > 0:
                table[(j, i)].get_text().set_color('green')
            if j == 0 and i == 0:
                table[(j, i)].get_text().set_color('black')

    total_avg = sum(avg_scores) / len(avg_scores)
    total_avg = int(total_avg) if total_avg % 1 < 0.5 else int(total_avg) + 1
    fig.suptitle("Total Average Score: $\\bf{}$%".format(total_avg), y=0.01, x=0.7)
    fig.tight_layout()
    plt.savefig("{}/overview/{}_categories.pdf".format(cfg.eval_dir, eval_name), bbox_inches='tight')
    plt.clf()


def plot_predictions_by_category_graph(outputs, categories, eval_name):
    levels_forcing = [0.0, 0.01, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3]
    norm_forcing = matplotlib.colors.BoundaryNorm(levels_forcing, 13)
    colors_forcing = ["#FFFFFF", "#FFFBAA", "#FFF88A", "#FFF56A", "#FFF24A", "#FFEF2A", "#FFEC0A", "#FFC408", "#FF9606",
                      "#FF6704", "#FF3802", "#800026"]

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
            class_predictions[cfg.label_names[outputs[i]]][cfg.val_categories.index(categories[i])] += 1.0 / len(
                cfg.val_samples)
        except ValueError:
            pass

    global_aod = import_forcing("/home/joe/PycharmProjects/climatestateclassifier/paper/tauttl.nc", "tauttl")[
                 12 * (years[0] - 1850):12 * (years[0] - 1850 + len(years) - (years[-1] - 1999))]
    # global_aod = import_forcing("/home/joe/PycharmProjects/climatestateclassifier/paper/tauttl.nc", "tauttl")
    global_mean_aod = np.nanmean(global_aod, axis=(1, 2))
    global_mean_aod = np.mean(global_mean_aod.reshape(-1, 12), axis=1)
    class_predictions["Global AOD"] = global_mean_aod

    # Calculate the width of each bar
    bar_width = 1

    font_size = 12
    matplotlib.rcParams.update({'font.size': font_size})

    # Create a figure and axis
    fig, ax = plt.subplots(nrows=2, figsize=(13, 4.5))
    fig.tight_layout()

    # Plot the first time series
    label_names = ["Southern Hemisphere", "Tropics", "Northern Hemisphere"]
    y_axes = ["SHE -", "TR -", "NHE -"]
    class_colors = ["gray", "red", "purple", "blue"]

    height = 0.1

    cmap_prob = matplotlib.cm.Greys
    current_bottom = 0.0
    for name, color in zip(label_names, class_colors):
        for year, value in zip(years, class_predictions[name]):
            img = ax[0].bar(year + 0.5, height, color=cmap_prob(value), width=bar_width, align='center',
                            bottom=current_bottom)
        current_bottom += height

    sm = ScalarMappable(cmap=cmap_prob)
    sm.set_array([])

    threshold = 0.005
    probabilities_nh = {}
    probabilities_tr = {}
    probabilities_sh = {}
    probabilities_ne = {}

    for i in range(class_predictions["Global AOD"].shape[0]):
        if class_predictions["Global AOD"][i] < threshold:
            probabilities_nh[years[i]] = int(
                math.ceil(100 * (100 * class_predictions["Northern Hemisphere"][i])) / 100) if math.ceil(
                100 * (100 * class_predictions["Northern Hemisphere"][i])) / 100 % 1 < 0.5 else int(math.ceil(
                100 * (100 * class_predictions["Northern Hemisphere"][i])) / 100) + 1
            probabilities_tr[years[i]] = int(
                math.ceil(100 * (100 * class_predictions["Tropics"][i])) / 100) if math.ceil(
                100 * (100 * class_predictions["Tropics"][i])) / 100 % 1 < 0.5 else int(math.ceil(
                100 * (100 * class_predictions["Tropics"][i])) / 100) + 1
            probabilities_sh[years[i]] = int(
                math.ceil(100 * (100 * class_predictions["Southern Hemisphere"][i])) / 100) if math.ceil(
                100 * (100 * class_predictions["Southern Hemisphere"][i])) / 100 % 1 < 0.5 else int(math.ceil(
                100 * (100 * class_predictions["Southern Hemisphere"][i])) / 100) + 1
            probabilities_ne[years[i]] = 100 - probabilities_nh[years[i]] - probabilities_tr[years[i]] - \
                                         probabilities_sh[years[i]]
            print(years[i])

    print(len(probabilities_nh.keys()))

    probabilities_nh["avg"] = sum(value for _, value in probabilities_nh.items()) / len(probabilities_nh.items())
    probabilities_tr["avg"] = sum(value for _, value in probabilities_tr.items()) / len(probabilities_tr.items())
    probabilities_sh["avg"] = sum(value for _, value in probabilities_sh.items()) / len(probabilities_sh.items())
    probabilities_ne["avg"] = sum(value for _, value in probabilities_ne.items()) / len(probabilities_ne.items())

    print(probabilities_nh["avg"])
    print(probabilities_tr["avg"])
    print(probabilities_sh["avg"])
    print(probabilities_ne["avg"])
    print(100 - probabilities_nh["avg"] - probabilities_tr["avg"] - probabilities_sh["avg"])
    cbar = plt.colorbar(sm, ax=ax[0], location="right", ticks=[0, 0.25, 0.5, 0.75, 1])

    volcanoes = {
        1883: ("Krakatau", "black"),
        1902: ("Santa Maria", "blue"),
        1912: ("Katmai", "yellow"),
        1963: ("Agung", "green"),
        1974: ("Fuego", "red"),
        1982: ("El Chichon", "gray"),
        1991: ("Pinatubo", "white")
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

    img = ax[1].imshow(np.flip(np.transpose(global_aod.squeeze()), axis=0), extent=[years[0], years[-1], -89, 89],
                       interpolation='nearest', aspect='auto', cmap=cmap_forcing, norm=norm_forcing)
    # ax[0].set_aspect(40)
    # ax[1].set_aspect(0.06)
    ax[1].yaxis.set_visible(False)
    ann_pos = [-6.1, 14.75, 58.28, -8.34, 14.47, 17.36, 15.13]

    i = 1
    for key, (name, color), pos in zip(volcanoes.keys(), volcanoes.values(), ann_pos):
        ax[1].text(key - 2.3, pos - 15, i, fontsize=font_size)
        ax[1].plot(key, pos, 'o', ms=7, mec='k', color="white")
        i += 1

    cbar = fig.colorbar(img, ax=ax[1], location="right", ticks=[0.0, 0.03, 0.07, 0.15, 0.3])
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
    # cmip6aod_mean = np.convolve(cmip6aod_mean, np.ones(12) / 12, mode='same')
    cmip6aod_mean = cmip6aod_mean[12 * (years[0] - 1850):12 * (years[0] - 1850 + len(years) - (years[-1] - 1999))]

    cmip5aod = import_forcing("/home/joe/PycharmProjects/climatestateclassifier/paper/tauttl.nc", "tauttl")
    cmip5aod_mean = np.nanmean(cmip5aod, axis=(1, 2))
    # cmip6aod_mean = np.convolve(cmip6aod_mean, np.ones(12) / 12, mode='same')
    cmip5aod_mean = cmip5aod_mean[12 * (years[0] - 1850):12 * (years[0] - 1850 + len(years) - (years[-1] - 1999))]

    glossacaod = import_forcing("/home/joe/PycharmProjects/climatestateclassifier/paper/GloSSAC_V2.0.nc",
                                "Glossac_Aerosol_Optical_Depth")
    glossacaod_mean = np.nanmean(glossacaod, axis=(1, 2))
    # glossacaod_mean = np.convolve(glossacaod_mean, np.ones(12) / 12, mode='same')
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
        for key, value in combined_predictions.items():
            total_numbers[i] += value[i]

    fig = plt.figure(figsize=(10, 2))

    color = cfg.timeseries_colors[0]
    ax1 = fig.add_axes([0, 0, 1, 1])

    ax2 = ax1.twinx()
    ax1.set_ylabel('Predicted Eruptions in %', color="black")

    ax1.bar([year + 0.5 for year in years],
            [((combined_predictions["Northern Hemisphere"][i]) / total_numbers[i]) * 100 if total_numbers[i] != 0 else 0
             for i in
             range(len(combined_predictions["No Eruption"]))],
            bottom=[((combined_predictions["Tropics"][i]) / total_numbers[i]) * 100 if total_numbers[
                                                                                           i] != 0 else 0
                    for i in range(len(combined_predictions["No Eruption"]))],
            color=cfg.timeseries_colors[0], label="NHE")

    ax1.bar([year + 0.5 for year in years],
            [((combined_predictions["Tropics"][i]) / total_numbers[i]) * 100 if total_numbers[i] != 0 else 0
             for i in range(len(combined_predictions["No Eruption"]))],
            bottom=[((combined_predictions["Southern Hemisphere"][i]) / total_numbers[i]) * 100 if total_numbers[
                                                                                                       i] != 0 else 0
                    for i in range(len(combined_predictions["No Eruption"]))],
            color=cfg.timeseries_colors[1], label="TR")

    ax1.bar([year + 0.5 for year in years],
            [((combined_predictions["Southern Hemisphere"][i]) / total_numbers[i]) * 100 if total_numbers[i] != 0 else 0
             for i in range(len(combined_predictions["No Eruption"]))],
            color=cfg.timeseries_colors[2], label="SHE")

    ax1.bar([year + 0.5 for year in [years[0] - 1] + years[:-1]],
            [0 for i in range(len(combined_predictions["No Eruption"]))])

    ax1.tick_params(axis='y', labelcolor="black")
    color = 'black'
    ax2.set_ylabel('Average Global AOD', color="black")  # we already handled the x-label with ax1
    # ax2.plot([years[0] + (1.0/12) * x for x in range(len(cmip5aod_mean))], [aod for aod in cmip5aod_mean], "y--", label="CMIP5 AOD")
    ax2.plot([years[0] + (1.0 / 12) * x for x in range(len(cmip6aod_mean))], [aod * 10 for aod in cmip6aod_mean], "k--",
             label="CMIP6 AOD")
    ax2.plot([1979 + (1.0 / 12) * x for x in range(len(glossacaod_mean))], glossacaod_mean, color='gray',
             linestyle='--', label="GloSSAC AOD")
    ax2.tick_params(axis='y', labelcolor="black")
    ax2.margins(x=0, y=0.01)
    ax1.margins(x=0)
    plt.figlegend(loc='upper center', fancybox=True, framealpha=1, shadow=True, fontsize='small',
                  bbox_to_anchor=(0.6, .99), ncol=2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Set the x-axis limits and labels
    plt.savefig("{}/{}_timeseries.pdf".format(cfg.eval_dir, eval_name), bbox_inches='tight')
    plt.clf()


def plot_predictions_by_category_graph_1800(outputs, categories, eval_name):
    levels_forcing = [0.0, 0.01, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3]
    norm_forcing = matplotlib.colors.BoundaryNorm(levels_forcing, 13)
    colors_forcing = ["#FFFFFF", "#FFFBAA", "#FFF88A", "#FFF56A", "#FFF24A", "#FFEF2A", "#FFEC0A", "#FFC408", "#FF9606",
                      "#FF6704", "#FF3802", "#800026"]

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
            class_predictions[cfg.label_names[outputs[i]]][cfg.val_categories.index(categories[i])] += 1.0 / len(
                cfg.val_samples)
        except ValueError:
            pass

    global_aod = import_forcing("/home/joe/PycharmProjects/climatestateclassifier/paper/eva_holo_total.nc",
                                "aod")[12*(years[0]-1800):12*(years[0]-1800 + len(years) - 1)]
    global_mean_aod = np.nanmean(global_aod, axis=(1, 2))
    global_mean_aod = np.mean(global_mean_aod.reshape(-1, 12), axis=1)
    class_predictions["Global AOD"] = global_mean_aod

    global_aod = global_aod[:, :, 9]

    # global_aod = np.flip(global_aod, axis=0)
    global_aod = np.flip(global_aod, axis=1)
    # global_aod = np.transpose(global_aod)

    # Calculate the width of each bar
    bar_width = 1

    font_size = 16
    matplotlib.rcParams.update({'font.size': font_size})

    # Create a figure and axis
    fig, ax = plt.subplots(nrows=2, figsize=(13, 4.5))
    fig.tight_layout()

    # ax[0].set_title("MPI-GE Member Classifications")
    # ax[1].set_title("Stratospheric Aerosol Optical Depth Field", loc="bottom")

    # Plot the first time series
    label_names = ["Southern Hemisphere", "Tropics", "Northern Hemisphere"]
    y_axes = ["SHE -", "TR -", "NHE -"]
    class_colors = ["gray", "red", "purple", "blue"]

    height = 0.1
    colors_prob = ["#FFFFFF", "#FFF0F0", "#FFE1E1", "#FFD2D2", "#FFC3C3", "#FFB4B4", "#FFA5A5", "#FF9696", "#FF8787",
                   "#FF7878", "#FF6969", "#FF0000"]

    cmap_prob = matplotlib.cm.Greys
    current_bottom = 0.0
    for name, color in zip(label_names, class_colors):
        for year, value in zip(years, class_predictions[name]):
            img = ax[0].bar(year + 0.5, height, color=cmap_prob(value), width=bar_width, align='center',
                            bottom=current_bottom)
        current_bottom += height

    levels_prob = [0.0, 0.08, 0.16, 0.25, 0.33, 0.41, 0.5, 0.58, 0.66, 0.75, 0.83, 0.91, 1.0]
    # norm_prob = matplotlib.colors.BoundaryNorm(levels_prob, 13)

    # cmap_prob = matplotlib.colors.ListedColormap(
    #    colors_prob)
    # cmap_prob = cmap_prob(np.arange(cmap_prob.N))
    # cmap_prob = ListedColormap(cmap_prob)

    sm = ScalarMappable(cmap=cmap_prob)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax[0], location="right", ticks=[0, 0.25, 0.5, 0.75, 1])

    volcanoes = {
        1809.4: ("Unknown", "magenta"),
        1815.4: ("Tambora", "magenta"),
        1822.8: ("Galunggung", "magenta"),
        1831: ("Babuyan Claro", "magenta"),
        1835: ("Cosigüina", "magenta"),
        1853.4: ("Toya", "magenta"),
        1856.8: ("Hokkaido-Komagatake", "magenta"),
        1861.9: ("Makian", "magenta"),
        1873: ("Grímsvötn", "magenta"),
        1875.4: ("Askja", "magenta"),
        1883.6: ("Krakatau", "magenta"),
        1886.5: ("Okataina", "magenta"),
        # 1902: ("Santa Maria", "blue"),
        # 1912: ("Katmai", "yellow"),
        # 1963: ("Agung", "green"),
        # 1974: ("Fuego", "red"),
        # 1982: ("El Chichon", "gray"),
        # 1991: ("Pinatubo", "white")
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

    ax[0].text(1809 - 3.2, -0.054, 1809, fontsize=font_size)
    ax[0].text(1809 - 1.4, 0.0, '-------------------', color="black", rotation=90)

    img = ax[1].imshow(np.flip(np.transpose(global_aod.squeeze()), axis=0), extent=[years[0], years[-1], -89, 89],
                       interpolation='nearest', aspect='auto', cmap=cmap_forcing, norm=norm_forcing)
    # ax[0].set_aspect(40)
    # ax[1].set_aspect(0.06)
    ax[1].yaxis.set_visible(False)
    # ann_pos = [-8.24, -7.25, 19.52, 12.97, -6.1, 14.75, 58.28, -8.34, 14.47, 17.36, 15.13]
    ann_pos = [20.0, -8.24, -7.25, 19.52, 12.97, 42.50, 42.06, 0.32, 64.40, 65.03, -6.1, -38.12]

    i = 1
    for key, (name, color), pos in zip(volcanoes.keys(), volcanoes.values(), ann_pos):
        if i == 1:
            ax[1].text(key - 2, pos - 28 - len(str(i)), i, fontsize=font_size)
            ax[1].text(key - 1.4, pos - 102, '------------------', color="red", rotation=90)
        else:
            ax[1].text(key - 2, pos - 28 - len(str(i)), i, fontsize=font_size)
            ax[1].plot(key, pos, 'o', ms=7, mec='k', color="white")
        i += 1

    cbar = fig.colorbar(img, ax=ax[1], location="right", ticks=[0.0, 0.03, 0.07, 0.15, 0.3])
    y_axes = ["50°S -", "0° -", "50°N -"]

    current_bottom = -50
    for name in y_axes:
        ax[1].text(years[0], current_bottom, name, ha='right', va='center')
        current_bottom += 50

    # Show the plot
    plt.savefig("{}/overview/{}_categories_graph.pdf".format(cfg.eval_dir, eval_name), bbox_inches='tight')
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

        if cfg.lons:
            dims["lon"] = dims["lon"][:cfg.lons]
        if cfg.lats:
            dims["lat"] = dims["lat"][:cfg.lats]

        # Plot inputs
        for time in range(inputs.shape[1]):
            ax[time, 0].axis('off')
            ax[time, 0].axis('tight')
            for var in range(inputs.shape[2]):
                vmin = -1.5  # torch.min(raw_input[i, var]) / 2
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
