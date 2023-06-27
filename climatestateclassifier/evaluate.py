import os

import torch

import lrp
from . import config as cfg
from .model.decoder import LocationDecoder
from .model.encoder import Encoder
from .model.net import LocationNet
from .utils.explain_location_net import generate_explanations
from .utils.io import load_ckpt
from .utils.netcdfloader import LocationNetCDFLoader
from .utils.plot_location import plot_single_predictions, plot_explanations, \
    plot_mean_explanations, plot_class_predictions, plot_ssi_predictions, plot_prediction_overview


def create_location_prediction(model, val_ensembles):
    dataset = LocationNetCDFLoader(cfg.data_root_dir, cfg.in_names, cfg.in_types, cfg.in_sizes, val_ensembles,
                                   cfg.val_ssis, cfg.locations, cfg.norm_to_ssi)
    data = []
    data_raw = []
    for i in range(4):
        data.append(torch.stack([dataset[j][i] for j in range(dataset.__len__())]))
        data_raw.append(torch.stack([dataset.__getitem__(j, raw_input=True)[i] for j in range(dataset.__len__())]))

    with torch.no_grad():
        predictions = model(data[0].to(cfg.device))

    dims = None
    explanations = None
    # get results from trained network
    if cfg.plot_single_explanations or cfg.plot_mean_explanations:
        # get lons and lats
        coords = dataset.xr_dss[1].coords
        dims = {}
        for dim in coords.dims:
            for key in ("time", "lon", "lat"):
                if key in dim:
                    dims[key] = coords[dim].values
        explanations = generate_explanations(model, data[0])

    predictions = predictions.to(torch.device('cpu'))
    return predictions, data[1].to(torch.device('cpu')), data[2], data[3], dims, explanations, data_raw[0]


def evaluate(arg_file=None, prog_func=None):
    cfg.set_evaluate_args(arg_file, prog_func)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    if not os.path.exists(cfg.eval_dir):
        os.makedirs(cfg.eval_dir)
    if not os.path.exists('{}/overview/'.format(cfg.eval_dir)):
        os.makedirs('{}/overview/'.format(cfg.eval_dir))
    if not os.path.exists('{}/total/'.format(cfg.eval_dir)):
        os.makedirs('{}/total/'.format(cfg.eval_dir))
    if not os.path.exists('{}/explanation/'.format(cfg.eval_dir)):
        os.makedirs('{}/explanation/'.format(cfg.eval_dir))
    if not os.path.exists('{}/explanation/mean/'.format(cfg.eval_dir)):
        os.makedirs('{}/explanation/mean/'.format(cfg.eval_dir))
    if not os.path.exists('{}/explanation/single/'.format(cfg.eval_dir)):
        os.makedirs('{}/explanation/single/'.format(cfg.eval_dir))

    n_models = len(cfg.model_names)
    assert n_models == len(cfg.eval_names)

    if cfg.mean_input:
        in_channels = len(cfg.in_names)
    else:
        in_channels = len(cfg.in_names) * cfg.time_steps

    for i_model in range(n_models):
        if cfg.val_ssis:
            if cfg.rotate_ensembles:
                # create rotation predictions
                predictions, labels, ssis, ensembles, explanations, inputs_raw = [], [], [], [], [], []
                for rotation in range(0, len(cfg.val_ensembles)):
                    rotation_string = 'rotation_{}'.format(rotation)
                    val_ensembles = set(cfg.val_ensembles[rotation:rotation + 1])
                    encoder = Encoder
                    decoder = LocationDecoder
                    model = LocationNet(encoder, decoder, img_size=cfg.in_sizes[0],
                                        in_channels=in_channels,
                                        encoding_layers=cfg.encoding_layers, stride=(1, 1), bn=False).to(cfg.device)
                    load_ckpt("{}/{}{}.pth".format(cfg.model_dir, cfg.model_names[i_model], rotation_string),
                              [('model', model)], cfg.device)
                    model.eval()
                    if cfg.plot_single_explanations or cfg.plot_mean_explanations:
                        model = lrp.converter.convert_location_net(model).to(cfg.device)
                    single_predictions, single_labels, single_ssis, single_ensembles, dims, single_explanation,\
                        input_raw = create_location_prediction(model, val_ensembles)
                    predictions.append(single_predictions)
                    labels.append(single_labels)
                    ssis.append(single_ssis)
                    ensembles.append(single_ensembles)
                    explanations.append(single_explanation)
                    inputs_raw.append(input_raw)
                predictions = torch.cat(predictions)
                labels = torch.cat(labels)
                ssis = torch.cat(ssis)
                ensembles = torch.cat(ensembles)
                if cfg.plot_single_explanations or cfg.plot_mean_explanations:
                    explanations = torch.cat(explanations, dim=1)
                inputs_raw = torch.cat(inputs_raw)
            else:
                encoder = Encoder
                decoder = LocationDecoder
                # create ssi predictions
                model = LocationNet(encoder, decoder, img_size=cfg.in_sizes[0],
                                    in_channels=in_channels,
                                    encoding_layers=cfg.encoding_layers, stride=(1, 1), bn=False,
                                    activation=False).to(cfg.device)
                load_ckpt("{}/{}.pth".format(cfg.model_dir, cfg.model_names[i_model]), [('model', model)], cfg.device)
                model.eval()
                if cfg.plot_single_explanations or cfg.plot_mean_explanations:
                    model = lrp.converter.convert_location_net(model).to(cfg.device)
                predictions, labels, ssis, ensembles, dims, explanations, \
                    inputs_raw = create_location_prediction(model, cfg.val_ensembles)

        if cfg.plot_prediction_overview:
            plot_prediction_overview(predictions, labels, eval_name="{}".format(cfg.eval_names[i_model]))
            plot_class_predictions(predictions, labels, eval_name="{}".format(cfg.eval_names[i_model]))
            plot_ssi_predictions(predictions, labels, ssis, eval_name="{}".format(cfg.eval_names[i_model]))
        if cfg.plot_single_predictions:
            plot_single_predictions(predictions, labels, ssis, ensembles, eval_name="{}".format(cfg.eval_names[i_model]))
        if cfg.plot_single_explanations:
            plot_explanations(inputs_raw, dims, labels, predictions, ensembles, ssis, explanations,
                              eval_name="{}".format(cfg.eval_names[i_model]))
        if cfg.plot_mean_explanations:
            plot_mean_explanations(inputs_raw, dims, labels, explanations, eval_name="{}".format(cfg.eval_names[i_model]))


if __name__ == "__main__":
    evaluate()
