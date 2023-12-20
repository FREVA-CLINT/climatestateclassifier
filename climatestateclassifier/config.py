import argparse
import os
import os.path

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
    arg_parser.add_argument('--data-types', type=str_list, default='tsurf',
                            help="Comma separated list of input variable types")
    arg_parser.add_argument('--device', type=str, default='cuda', help="Device used by PyTorch (cuda or cpu)")
    arg_parser.add_argument('--normalization', type=str, default=None,
                            help="None: No normalization, "
                                 "std: normalize to 0 mean and 1 std, "
                                 "img: normalize values between -1 and 1 and 0.5 mean and 0.5 std, "
                                 "custom: normalize with custom define mean and std values")
    arg_parser.add_argument('--val-samples', type=interv_list, default='101,200',
                            help="Comma separated list of samples that are used for validation")
    arg_parser.add_argument('--val-categories', type=str_list, default=None,
                            help="Comma separated list of category values that are used for validation")
    arg_parser.add_argument('--attention-dim', type=int, default=None,
                            help="Dimension of attention layer")
    arg_parser.add_argument('--decoder-dims', type=int_list, default="512,64",
                            help="Comma separated list of dimensions of decoding layer")
    arg_parser.add_argument('--encoder-dims', type=int_list, default="16,32,64",
                            help="Comma separated list of dimensions of encoding layers")
    arg_parser.add_argument('--loss-criterion', type=str, default="ce",
                            help="Loss criterion: Cross-Entropy (ce), Mean Square Error (mse) or L1 (l1)")
    arg_parser.add_argument('--rotate-samples', action='store_true',
                            help="Rotates the training cycle through all samples. In each cycle,"
                                 " a single sample is left out for validation, all others are used for training")
    arg_parser.add_argument('--max-rotations', type=int, default=5, help="Stop rotations after specified number")
    arg_parser.add_argument('--time-steps', type=int, default=1, help="Number of time steps in each sample")
    arg_parser.add_argument('--vlim', type=int_list, default="-1,1",
                            help="Comma separated list of vmin, vmax values for the color scale of the snapshot graphs")
    arg_parser.add_argument('--labels', type=str_list, default=',nh,sh,ne',
                            help="Comma separated list of labels for classifier")
    arg_parser.add_argument('--label-names', type=str_list, default='Tropics,Northern Hemisphere,Southern Hemisphere,'
                                                                    'No Eruption',
                            help="Comma separated list of labels for classifier")
    arg_parser.add_argument('--random-seed', type=int, default=None,
                            help="Random seed for iteration loop and initialization weights")
    arg_parser.add_argument('--global-padding', action='store_true', help="Use a custom padding for global dataset")
    arg_parser.add_argument('--mean-input', action='store_true', help="Use a custom padding for global dataset")
    arg_parser.add_argument('--lazy-load', action='store_true', help="Load data sets during training")
    arg_parser.add_argument('--activation_out', type=str, default='',
                            help="Output activation")

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
    arg_parser.add_argument('--train-categories', type=str_list, default='0,5,10,20,40',
                            help="Comma separated list of category values that are used for training")
    arg_parser.add_argument('--train-samples', type=interv_list, default='101',
                            help="Comma separated list of samples that are used for training")
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
    arg_parser.add_argument('--plot-overview', action='store_true', help="Create overview tables")
    arg_parser.add_argument('--plot-tables', action='store_true', help="Plot prediction overview")
    arg_parser.add_argument('--plot-explanations', action='store_true', help="Plot LRP explanations")
    arg_parser.add_argument('--explanation-cmap', type=str_list,
                            default="#FFFFFF,#E6F2E6,#CCE5CC,#B3D9B3,#99CC99,#80BF80,#006600",
                            help="Comma separated list of colors for explanation cmap")
    arg_parser.add_argument('--explanation-names', type=str_list, default="gamma",
                            help="Comma separated list of explanations. Choose from: "
                                 "gradient, epsilon, gamma, gamma+epsilon, alpha1beta0, alpha2beta1, "
                                 "patternattribution, patternnet")
    global_args(arg_parser, arg_file, prog_func)
