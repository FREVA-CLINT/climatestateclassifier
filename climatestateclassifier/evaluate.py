import os

import torch

from .lrp import converter
from . import config as cfg
from .model.net import ClassificationNet
from .utils.explain_net import generate_explanations
from .utils.io import load_ckpt
from .utils.netcdfloader import NetCDFLoader
from .utils.plot_utils import plot_single_predictions, plot_explanations, \
    plot_class_predictions, plot_predictions_by_category, plot_prediction_overview, plot_predictions_by_category_graph, \
    read_results_from_csv, save_results_as_csv, plot_predictions_by_category_timeseries, \
    plot_predictions_by_category_graph_1800


def create_prediction(model_name, val_samples):
    # load data
    dataset = NetCDFLoader(cfg.data_root_dirs, cfg.data_types, val_samples, cfg.val_categories, cfg.labels)
    input = torch.stack([dataset[j][0] for j in range(dataset.__len__())]).to(torch.device('cpu'))
    label = torch.stack([dataset[j][1] for j in range(dataset.__len__())]).to(torch.device('cpu'))
    category = [dataset[j][2] for j in range(dataset.__len__())]
    sample_name = [dataset[j][3] for j in range(dataset.__len__())]

    # create model and predictions
    in_channels = len(cfg.data_types) if cfg.mean_input else len(cfg.data_types) * cfg.time_steps
    model = ClassificationNet(img_sizes=dataset.img_sizes[0],
                              in_channels=in_channels,
                              enc_dims=[dim for dim in cfg.encoder_dims],
                              dec_dims=[dim for dim in cfg.decoder_dims],
                              n_classes=len(cfg.labels)).to(cfg.device)

    load_ckpt(model_name, [('model', model)], cfg.device)
    model.eval()
    if cfg.plot_explanations:
        model = converter.convert_net(model).to(cfg.device)
    with torch.no_grad():
        output = model(input.to(cfg.device)).to(torch.device('cpu'))

    dims = None
    explanations = None
    # get results from trained network
    if cfg.plot_explanations:
        coords = dataset.xr_dss[1].coords
        dims = {}
        for dim in coords.dims:
            for key in ("time", "lon", "lat"):
                if key in dim:
                    dims[key] = coords[dim].values
        explanations = generate_explanations(model, input.to(cfg.device)).to(torch.device('cpu'))

    # renormalize input data
    input = torch.stack(torch.split(input, len(cfg.data_types), dim=1), dim=1)
    if cfg.normalization:
        for v in range(len(cfg.data_types)):
            for i in range(input.shape[0]):
                input[i, :, v, :, :] = dataset.data_normalizer.renormalize(input[i, :, v, :, :], v)

    return input, output, label, category, sample_name, dims, explanations


def evaluate(arg_file=None, prog_func=None):
    cfg.set_evaluate_args(arg_file, prog_func)

    if cfg.plot_timeseries:
        outputs = {}
        categories = {}
        for name in cfg.timeseries_names:
            _, outputs[name], categories[name], _ = read_results_from_csv("{}/{}".format(cfg.eval_dir, name),
                                                                          cfg.eval_names[0])
        plot_predictions_by_category_timeseries(outputs, categories, eval_name="{}".format(cfg.eval_names[0]))
    else:

        if not os.path.exists(cfg.log_dir):
            os.makedirs(cfg.log_dir)

        if not os.path.exists(cfg.eval_dir):
            os.makedirs(cfg.eval_dir)
        if not os.path.exists('{}/overview/'.format(cfg.eval_dir)):
            os.makedirs('{}/overview/'.format(cfg.eval_dir))
        if not os.path.exists('{}/total/'.format(cfg.eval_dir)):
            os.makedirs('{}/total/'.format(cfg.eval_dir))
        if not os.path.exists('{}/explanations/'.format(cfg.eval_dir)):
            os.makedirs('{}/explanations/'.format(cfg.eval_dir))

        n_models = len(cfg.model_names)

        for i_model in range(n_models):
            if cfg.load_from_csv:
                labels, outputs, categories, sample_names = read_results_from_csv(cfg.eval_dir, cfg.eval_names[i_model])
            else:
                if cfg.rotate_samples:
                    # create rotation predictions
                    inputs, outputs, labels, categories, sample_names, explanations = [], [], [], [], [], []
                    for rotation in range(0, len(cfg.val_samples)):
                        val_samples = set(cfg.val_samples[rotation:rotation + 1])
                        model_name = "{:s}/ckpt/{:s}{:s}.pth".format(
                            cfg.model_dir, cfg.model_names[i_model], 'rotation_{}'.format(rotation))

                        input, output, label, category, sample_name, dims, explanation = create_prediction(model_name, val_samples)
                        inputs.append(input)
                        outputs.append(output)
                        labels.append(label)
                        explanations.append(explanation)
                        categories += category
                        sample_names += sample_name
                    inputs = torch.cat(inputs)
                    outputs = torch.cat(outputs)
                    labels = torch.cat(labels)
                    if cfg.plot_explanations:
                        explanations = torch.cat(explanations, dim=1)
                else:
                    # create normal predictions
                    model_name = "{:s}/ckpt/{:s}.pth".format(cfg.model_dir, cfg.model_names[i_model])
                    inputs, outputs, labels, categories, sample_names, dims, explanations = create_prediction(
                        model_name, cfg.val_samples)
                outputs = outputs.argmax(1)
                labels = labels.argmax(1)
                save_results_as_csv(cfg.eval_dir, cfg.eval_names[i_model], labels, outputs, categories, sample_names)
                if cfg.plot_explanations:
                    #plot_explanations(inputs, dims, labels, outputs, sample_names, categories, explanations,
                    #                  eval_name="{}".format(cfg.eval_names[i_model]))
                    plot_explanations(torch.mean(inputs, dim=0).unsqueeze(0),
                                      dims,
                                      labels[0].unsqueeze(0),
                                      outputs[0].unsqueeze(0),
                                      [sample_names[0]],
                                      [categories[0]],
                                      torch.mean(explanations, dim=0).unsqueeze(0),
                                      eval_name="average_{}".format(cfg.eval_names[i_model]))

            if cfg.plot_prediction_overview:
                #plot_prediction_overview(outputs, labels, eval_name="{}".format(cfg.eval_names[i_model]))
                #plot_class_predictions(outputs, labels, eval_name="{}".format(cfg.eval_names[i_model]))
                #plot_predictions_by_category(outputs, labels, categories, eval_name="{}".format(cfg.eval_names[i_model]))
                plot_predictions_by_category_graph_1800(outputs, categories, eval_name="{}".format(cfg.eval_names[i_model]))
            if cfg.plot_single_predictions:
                print(categories)
                plot_single_predictions(outputs, labels, categories, sample_names, eval_name="{}".format(cfg.eval_names[i_model]))


if __name__ == "__main__":
    evaluate()
