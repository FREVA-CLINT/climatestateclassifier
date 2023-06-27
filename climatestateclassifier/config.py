import argparse
import os
import os.path

LAMBDA_DICT_ONI = {
    'graph': 1.0
}

LAMBDA_DICT_LOCATION = {
    'class': 1.0
}


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.parse_args(open(values).read().split(), namespace)


def str_list(arg):
    return arg.split(',')


def int_list(arg):
    return list(map(int, arg.split(',')))


def float_list(arg):
    return list(map(float, arg.split(',')))


def lim_list(arg):
    lim = list(map(float, arg.split(',')))
    assert len(lim) == 2
    return lim


def interv_list(arg):
    interv_list = []
    for interv in arg.split(','):
        if "-" in interv:
            intervals = interv.split("-")
            interv_list += range(int(intervals[0]), int(intervals[1]) + 1)
        else:
            interv_list.append(int(interv))
    return interv_list


def global_args(parser, arg_file=None, prog_func=None):
    import torch

    if arg_file is None:
        import sys
        argv = sys.argv[1:]
    else:
        argv = ["--load-from-file", arg_file]

    global progress_fwd
    progress_fwd = prog_func

    args = parser.parse_args(argv)

    args_dict = vars(args)
    for arg in args_dict:
        globals()[arg] = args_dict[arg]

    torch.backends.cudnn.benchmark = True
    globals()[device] = torch.device(device)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def set_common_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/',
                            help="Root directory containing the climate datasets")
    arg_parser.add_argument('--log-dir', type=str, default='logs/', help="Directory where the log files will be stored")
    arg_parser.add_argument('--in-names', type=str_list, default='pr,temp2,slp,tsurf',
                            help="Comma separated list of netCDF files (input dataset)")
    arg_parser.add_argument('--out-names', type=str_list, default='tsurf',
                            help="Comma separated list of netCDF files (gt dataset). ")
    arg_parser.add_argument('--in-types', type=str_list, default='pr,temp2,slp,tsurf',
                            help="Comma separated list of input variable types")
    arg_parser.add_argument('--in-sizes', type=int_list, default='192,192,192,192',
                            help="Comma separated list of input sizes")
    arg_parser.add_argument('--out-types', type=str_list, default='tsurf',
                            help="Comma separated list of output variable types")
    arg_parser.add_argument('--out-sizes', type=int_list, default='1',
                            help="Comma separated list of output sizes")
    arg_parser.add_argument('--device', type=str, default='cuda', help="Device used by PyTorch (cuda or cpu)")
    arg_parser.add_argument('--normalization', type=str, default=None,
                            help="None: No normalization, "
                                 "std: normalize to 0 mean and 1 std, "
                                 "img: normalize values between -1 and 1 and 0.5 mean and 0.5 std, "
                                 "custom: normalize with custom define mean and std values")
    arg_parser.add_argument('--val-ensembles', type=interv_list, default='101,200',
                            help="Comma separated list of ensembles that are used for validation")
    arg_parser.add_argument('--val-ssis', type=float_list, default=None,
                            help="Comma separated list of ssi values that are used for validation")
    arg_parser.add_argument('--val-colors', type=str_list, default=None,
                            help="Comma separated list of colors for plotting evaluation graphs")
    arg_parser.add_argument('--prediction-range', type=int, default=30,
                            help="Number of months to predict oni time series")
    arg_parser.add_argument('--prediction-index', type=int, default=None,
                            help="Use an index of the oni time series as ground truth")
    arg_parser.add_argument('--prediction-mean', action='store_true',
                            help="Use the mean oni value as ground truth")
    arg_parser.add_argument('--oni-range', type=lim_list, default="-4,6",
                            help="Number of months to predict oni time series")
    arg_parser.add_argument('--time-steps', type=int, default=1,
                            help="Number of time steps that are used as input")
    arg_parser.add_argument('--oni-resolution', type=int, default=20,
                            help="Resolution of oni value when using ce loss")
    arg_parser.add_argument('--attention-dim', type=int, default=None,
                            help="Dimension of attention layer")
    arg_parser.add_argument('--decoder-dims', type=int_list, default=[512, 64],
                            help="Dimension of decoding layer")
    arg_parser.add_argument('--encoding-layers', type=int, default=6,
                            help="Number of encoding layers")
    arg_parser.add_argument('--dropout', type=float, default=0.1,
                            help="Dropout probability for decoder")
    arg_parser.add_argument('--norm-to-ssi', type=float, default=None,
                            help="Dropout probability for decoder")
    arg_parser.add_argument('--lstm', action='store_true',
                            help="Use LSTM for decoding oni sequence")
    arg_parser.add_argument('--add-ssi', action='store_true',
                            help="Add ssi value to input of the decoder")
    arg_parser.add_argument('--loss-criterion', type=str, default="ce",
                            help="Loss criterion: Cross-Entropy (ce), Mean Square Error (mse) or L1 (l1)")
    arg_parser.add_argument('--rotate-ensembles', action='store_true',
                            help="Rotates the training cycle through all ensembles. In each cycle,"
                                 " a single ensemble is left out for validation, all others are used for training")
    arg_parser.add_argument('--reference-ssis', type=float_list, default=None,
                            help="Comma separated list of ssis values that are used"
                                 " for plotting the ground truth oni series")
    arg_parser.add_argument('--max-rotations', type=int, default=5, help="Stop rotations after specified number")
    arg_parser.add_argument('--beam-size', type=int, default=5, help="Size of beam for explanation beam")
    arg_parser.add_argument('--reference-colors', type=str_list, default=None,
                            help="Comma separated list of colors for plotting reference graphs")
    arg_parser.add_argument('--add-noise', action='store_true',
                            help="Add random noise to input data to reduce over-fitting")
    arg_parser.add_argument('--vlim', type=int_list, default="-1,3",
                            help="Comma separated list of vmin, vmax values for the color scale of the snapshot graphs")
    arg_parser.add_argument('--attention', action='store_true', help="Apply attention layer")
    arg_parser.add_argument('--reverse-jja-indices', type=int_list, default='17,18,19',
                            help="Create plot images of the results for the comma separated list of time indices")
    arg_parser.add_argument('--locations', type=str_list, default=',nh,sh,ne',
                            help="Comma separated list of classes for location prediction")
    arg_parser.add_argument('--random-seed', type=int, default=None,
                            help="Random seed for iteration loop and initialization weights")
    arg_parser.add_argument('--global-padding', action='store_true', help="Use a custom padding for global dataset")
    arg_parser.add_argument('--mean-input', action='store_true', help="Use a custom padding for global dataset")
    arg_parser.add_argument('--experiment', type=str, default=None, help="Read data via freva experiment")
    arg_parser.add_argument('--train-years', type=int_list, default='1992',
                            help="Comma separated list of ssi values that are used for training")
    return arg_parser


def set_train_args(arg_file=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/',
                            help="Parent directory of the training checkpoints and the snapshot images")
    arg_parser.add_argument('--resume-iter', type=int, help="Iteration step from which the training will be resumed")
    arg_parser.add_argument('--resume-rotation', type=int, default=0,
                            help="Rotation step from which the training will be resumed")
    arg_parser.add_argument('--batch-size', type=int, default=18, help="Batch size")
    arg_parser.add_argument('--n-threads', type=int, default=4, help="Number of threads")
    arg_parser.add_argument('--multi-gpus', action='store_true', help="Use multiple GPUs, if any")
    arg_parser.add_argument('--finetune', action='store_true',
                            help="Enable the fine tuning mode (use fine tuning parameterization "
                                 "and disable batch normalization")
    arg_parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    arg_parser.add_argument('--lr-finetune', type=float, default=5e-5, help="Learning rate for fine tuning")
    arg_parser.add_argument('--max-iter', type=int, default=1000000, help="Maximum number of iterations")
    arg_parser.add_argument('--log-interval', type=int, default=None,
                            help="Iteration step interval at which a tensorboard summary log should be written")
    arg_parser.add_argument('--lr-scheduler-patience', type=int, default=None, help="Patience for the lr scheduler")
    arg_parser.add_argument('--plot-snapshot-figures', action='store_true',
                            help="Save evaluation figure of the given index list for the iteration steps defined in "
                                 "--log-interval")
    arg_parser.add_argument('--save-model-interval', type=int, default=50000,
                            help="Iteration step interval at which the model should be saved")
    arg_parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile,
                            help="Load all the arguments from a text file")
    arg_parser.add_argument('--train-ssis', type=float_list, default='0,5,10,20,40',
                            help="Comma separated list of ssi values that are used for training")
    arg_parser.add_argument('--train-ensembles', type=interv_list, default='101',
                            help="Comma separated list of ensembles that are used for training")
    global_args(arg_parser, arg_file)


def set_evaluate_args(arg_file=None, prog_func=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--model-dir', type=str, default='snapshots/ckpt/', help="Directory of the trained models")
    arg_parser.add_argument('--model-names', type=str_list, default='1000000.pth', help="Model names")
    arg_parser.add_argument('--eval-dir', type=str, default='evaluation/',
                            help="Directory where the output files will be stored")
    arg_parser.add_argument('--eval-names', type=str_list, default='output',
                            help="Prefix used for the output filenames")
    arg_parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile,
                            help="Load all the arguments from a text file")
    arg_parser.add_argument('--std-percentile', type=float, default=0.0,
                            help="Percentile for creating evaluation graphs of oni time-series")
    arg_parser.add_argument('--action', type=str, default='test', help="Prediction action: evaluate (test) with "
                                                                       "ground truth or predict (predict) without "
                                                                       "ground truth")
    arg_parser.add_argument('--mm', type=int, default=None, help="Create plots with monthly mean")
    arg_parser.add_argument('--plot-overview', action='store_true', help="Create overview plots")
    arg_parser.add_argument('--plot-all-ensembles', action='store_true', help="Create plots with all ensembles")
    arg_parser.add_argument('--plot-single-ensembles', action='store_true',
                            help="Create single plots for each ensemble")
    arg_parser.add_argument('--plot-differences', action='store_true',
                            help="Create plots of differences for predicted"
                                 " ensembles compared to reference ensembles")
    arg_parser.add_argument('--plot-heatmaps', action='store_true', help="Create heatmap plots")
    arg_parser.add_argument('--plot-enso-tables', action='store_true', help="Create enso tables")
    arg_parser.add_argument('--norm-channels', action='store_true', help="Normalize LRP explanation over channels")
    arg_parser.add_argument('--color-plot', action='store_true', help="Normalize LRP explanation over channels")
    arg_parser.add_argument('--plot-prediction-overview', action='store_true', help="Normalize LRP explanation over channels")
    arg_parser.add_argument('--plot-single-predictions', action='store_true', help="Normalize LRP explanation over channels")
    arg_parser.add_argument('--plot-mean-explanations', action='store_true', help="Normalize LRP explanation over channels")
    arg_parser.add_argument('--plot-single-explanations', action='store_true', help="Normalize LRP explanation over channels")
    arg_parser.add_argument('--cmap-colors', type=str_list, default=None,
                            help="Comma separated list of classes for location prediction")
    arg_parser.add_argument('--explanation-names', type=str_list, default="gamma",
                            help="Comma separated list of explanations. Choose from: "
                                 "gradient, epsilon, gamma, gamma+epsilon, alpha1beta0, alpha2beta1, "
                                 "patternattribution, patternnet")
    arg_parser.add_argument('--eval-years', type=str_list, default="1992", help="Read data via freva experiment")
    arg_parser.add_argument('--gt-locations', type=str_list, default='', help="Read data via freva experiment")
    global_args(arg_parser, arg_file, prog_func)
