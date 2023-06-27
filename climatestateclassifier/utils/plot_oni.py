import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from .evaluation import prob_to_oni, get_references, calculate_mm_data, get_enso_probabilities
from .. import config as cfg


def plot_ensembles(data, label, alpha=1):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j == 0:
                ax = plt.gca()
                ax.set_ylim([cfg.vlim[0], cfg.vlim[1]])
                plt.plot(range(cfg.prediction_range), data[i][j], cfg.val_colors[i],
                         label="{} {}".format(label, cfg.val_ssis[i]), alpha=alpha)
            else:
                ax = plt.gca()
                ax.set_ylim([cfg.vlim[0], cfg.vlim[1]])
                plt.plot(range(cfg.prediction_range), data[i][j], cfg.val_colors[i], alpha=alpha)


def plot_enso_table(ssi_labels, enso_plus_list, enso_minus_list):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    d = {
        '$\\bf{SSI}$': reversed(ssi_labels),
        '$\\bf{ENSO+}$': reversed(enso_plus_list),
        '$\\bf{ENSO-}$': reversed(enso_minus_list)
    }
    # set colors
    colors = []
    for i in range(len(ssi_labels)):
        if i % 2 == 0:
            colors.append(["#caccce", "#caccce", "#caccce"])
        else:
            colors.append(["#e6e7e9", "#e6e7e9", "#e6e7e9"])

    df = pd.DataFrame(data=d)
    table = ax.table(colWidths=[0.1, 0.2, 0.2], cellText=df.values, colLabels=df.columns, loc='center', colColours=["#002f4a", "#002f4a", "#002f4a"], cellColours=colors)
    table[(0,0)].get_text().set_color('white')
    table[(0,1)].get_text().set_color('white')
    table[(0,2)].get_text().set_color('white')
    fig.tight_layout()


def plot_snapshot_figures(gt, output, filename, val_ensembles):
    output = output.to(torch.device('cpu'))
    if cfg.loss_criterion == 'ce':
        output = prob_to_oni(output)
    output = output
    gt = gt.to(torch.device('cpu'))

    if cfg.loss_criterion != 'ce':
        output = output.squeeze(1)

    data_list = [gt, output]
    labels = ['EVA values', 'Prediction']

    fig, ax = plt.subplots(nrows=len(cfg.val_ssis), sharey=True, sharex=True, figsize=(10, len(cfg.val_ssis) * 10))

    # plot and save data
    for graph in range(len(data_list)):
        for i in range(len(cfg.val_ssis)):
            for j in range(len(val_ensembles)):
                if len(cfg.val_ssis) > 1:
                    ax[i].plot(range(cfg.prediction_range), data_list[graph][i], label=labels[graph])
                    ax[i].set_title("ONI of ensemble {} with SSI={}".format(val_ensembles[j], cfg.val_ssis[i]))
                    ax[i].legend(labels, loc="upper left")
                else:
                    ax.plot(range(cfg.prediction_range), data_list[graph][j], label=labels[graph])
                    ax.set_title("ONI of ensemble {} with SSI={}".format(val_ensembles[j], cfg.val_ssis[i]))
                    ax.legend(labels, loc="upper left")
    plt.xlabel("Months after eruption")
    plt.ylabel("ONI index")

    plt.savefig(filename + '.jpg', dpi=300)
    plt.clf()
    plt.close('all')


def plot_evaluation_figures(output, heatmaps, inputs, eval_name):
    if output is not None:
        output = output.to(torch.device('cpu'))
        if cfg.loss_criterion == 'ce':
            new_output = []
            for i in range(output.shape[0]):
                new_output.append(prob_to_oni(output[i]))
            output = torch.stack(new_output, dim=0)

    if cfg.reference_ssis:
        references = get_references()

    if output is not None and cfg.mm:
        output = calculate_mm_data(output)

    # plot overview with min, max and mean
    if cfg.plot_overview:
        file_path = "{}/overview".format(cfg.eval_dir)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if output is not None:
            for i in range(output.shape[0]):
                output_mean = torch.mean(output[i], dim=0)
                output_min = np.percentile(output[i], 100 - cfg.std_percentile, axis=0)
                output_max = np.percentile(output[i], 0.0 + cfg.std_percentile, axis=0)
                ax = plt.gca()
                ax.set_ylim([cfg.vlim[0], cfg.vlim[1]])
                # plot mean, min and max ranges
                plt.plot(range(cfg.prediction_range), output_mean, '{}--'.format(cfg.val_colors[i]), label="Predicted Mean SSI{}".format(cfg.val_ssis[i]))
                plt.fill_between(range(cfg.prediction_range), output_min, output_max, color=cfg.val_colors[i], alpha=0.2)
        if cfg.reference_ssis:
            for i in range(references.shape[0]):
                gt_mean = torch.mean(references[i], dim=0)
                gt_max = np.percentile(references[i], 100 - cfg.std_percentile, axis=0)
                gt_min = np.percentile(references[i], 0.0 + cfg.std_percentile, axis=0)
                ax = plt.gca()
                ax.set_ylim([cfg.vlim[0], cfg.vlim[1]])
                # plot mean, min and max ranges
                plt.plot(range(cfg.prediction_range), gt_mean, '{}--'.format(cfg.reference_colors[i]), label="EVA Mean SSI{}".format(cfg.reference_ssis[i]))
                plt.fill_between(range(cfg.prediction_range), gt_min, gt_max, color=cfg.reference_colors[i], alpha=0.2)
        plt.legend(loc="upper right", prop={'size': 7})
        plt.xlabel("Months after eruption")
        plt.ylabel("ONI index")
        plt.savefig("{}/{}.jpg".format(file_path, eval_name), dpi=300)
        plt.clf()

    # plot ensembles
    if cfg.plot_all_ensembles:
        file_path = "{}/ensembles".format(cfg.eval_dir)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if output is not None:
            plot_ensembles(output, "Predicted SSI")
        if cfg.reference_ssis:
            plot_ensembles(references, "EVA SSI", alpha=0.4)
        plt.legend(loc="upper right", prop={'size': 7})
        plt.xlabel("Months after eruption")
        plt.ylabel("ONI index")
        plt.savefig("{}/{}.jpg".format(file_path, eval_name), dpi=300)
        plt.clf()

    # plot single ensembles
    if cfg.plot_single_ensembles:
        assert output is not None and references is not None and output.shape[0] == references.shape[0]
        file_path = "{}/single_ensembles".format(cfg.eval_dir)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                ax = plt.gca()
                ax.set_ylim([cfg.vlim[0], cfg.vlim[1]])
                plt.plot(range(cfg.prediction_range), output[i][j], cfg.val_colors[i],
                         label="Prediction SSI{}".format(cfg.val_ssis[i]))
                plt.plot(range(cfg.prediction_range), references[i][j], cfg.reference_colors[i],
                         label="EVA SSI{}".format(cfg.reference_ssis[i]))
                plt.legend(loc="upper right", prop={'size': 7})
                plt.xlabel("Months after eruption")
                plt.ylabel("ONI index")
                plt.savefig('{}/{}_ensemble{}_ssi{}.jpg'.format(file_path, eval_name, cfg.val_ensembles[j],
                                                                cfg.val_ssis[i]),
                            dpi=300)
                plt.clf()

    # plot ensemble differences
    if cfg.plot_differences:
        assert output is not None and references is not None and output.shape[0] == references.shape[0]
        file_path = "{}/differences".format(cfg.eval_dir)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if j == 0:
                    ax = plt.gca()
                    ax.set_ylim([cfg.vlim[0], cfg.vlim[1]])
                    plt.plot(range(cfg.prediction_range), output[i][j] - references[i][j], cfg.reference_colors[i],
                             label="Differences SSI{}-SSI{}".format(cfg.val_ssis[i], cfg.reference_ssis[i]),
                             alpha=0.2)
                else:
                    ax = plt.gca()
                    ax.set_ylim([cfg.vlim[0], cfg.vlim[1]])
                    plt.plot(range(cfg.prediction_range), output[i][j] - references[i][j], cfg.reference_colors[i],
                             alpha=0.2)
            plt.plot(range(cfg.prediction_range), torch.mean(torch.abs(references[i] - output[i]), dim=0),
                     '{}--'.format(cfg.val_colors[i]),
                     label="Mean Difference SSI{}-SSI{}".format(cfg.val_ssis[i], cfg.reference_ssis[i]))
        plt.legend(loc="upper right", prop={'size': 7})
        plt.xlabel("Months after eruption")
        plt.ylabel("ONI index")
        plt.savefig("{}/{}.jpg".format(file_path, eval_name), dpi=300)
        plt.clf()

    # plot heatmaps
    if cfg.plot_heatmaps:
        file_path = "{}/heatmaps".format(cfg.eval_dir)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                fig, ax = plt.subplots(nrows=inputs.shape[2], ncols=3, figsize=(3 * 10, inputs.shape[2] * 10))
                for c in range(inputs.shape[2]):
                    image = ax[c, 0].imshow(inputs[i, j, c])
                    heatmap = ax[c, 1].imshow(heatmaps[i, j, c])
                    fig.colorbar(heatmap, ax=ax[c, 1], location='right', anchor=(0, 0.3), shrink=0.7)
                    ax[c, 2].plot(range(cfg.prediction_range), output[i, j], label="Prediction SSI{}".format(cfg.val_ssis[i]))
                    if cfg.reference_ssis:
                        ax[c, 2].plot(range(cfg.prediction_range), references[i, j], label="EVA SSI{}".format(cfg.reference_ssis[i]))
                plt.savefig("{}/{}_ensemble_{}.jpg".format(file_path, eval_name, cfg.val_ensembles[j]), dpi=100)
                plt.clf()

    # plot enso tables
    if cfg.plot_enso_tables:
        file_path = "{}/enso_table".format(cfg.eval_dir)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if output is not None:
            enso_plus, enso_minus = get_enso_probabilities(output)
            plot_enso_table(cfg.val_ssis, enso_plus, enso_minus)
            plt.savefig("{}/{}_predicted.jpg".format(file_path, eval_name), dpi=300)
            plt.clf()
        if references is not None:
            enso_plus, enso_minus = get_enso_probabilities(references)
            plot_enso_table(cfg.reference_ssis, enso_plus, enso_minus)
            plt.savefig("{}/{}_eva.jpg".format(file_path, eval_name), dpi=300)
            plt.clf()

    plt.close('all')
