import math

import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
from matplotlib.colors import ListedColormap

from .. import config as cfg


def plot_prediction_overview(outputs, labels, eval_name):
    # Clean predictions
    indices = outputs.argmax(1)
    cleaned_predictions = torch.zeros(outputs.shape).scatter(1, indices.unsqueeze(1), 1.0)
    labels = labels.int()

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
    pred_indices = outputs.argmax(1)
    gt_indices = labels.argmax(1)

    prediction_labels = []
    gt_labels = []
    prediction_colors = []
    for i in range(pred_indices.shape[0]):
        gt_labels.append(cfg.label_names[gt_indices[i]])
        prediction_labels.append(cfg.label_names[pred_indices[i]])
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


def plot_class_predictions(predictions, labels, eval_name):
    # Clean predictions
    pred_indices = predictions.argmax(1)
    gt_indices = labels.argmax(1)

    class_predictions = [[0 for j in range(len(cfg.labels))] for i in range(len(cfg.labels))]
    for i in range(pred_indices.shape[0]):
        class_predictions[gt_indices[i]][pred_indices[i]] += 1

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


def plot_predictions_by_category(predictions, labels, categories, eval_name):
    # Clean predictions
    pred_indices = predictions.argmax(1)
    gt_indices = labels.argmax(1)

    class_predictions = [[0 for j in range(len(cfg.labels) * len(cfg.val_categories))] for i in range(len(cfg.labels))]
    for i in range(len(cfg.val_categories)):
        for k in range(gt_indices.shape[0]):
            if cfg.val_categories[i] == categories[k]:
                class_predictions[gt_indices[k]][pred_indices[k] + (i * len(cfg.labels))] += (
                    (1.0 / len(cfg.val_samples)) if categories[k] != 0.0 else 1.0 / len(cfg.val_samples))

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
