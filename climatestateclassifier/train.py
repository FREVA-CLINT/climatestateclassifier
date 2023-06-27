import os

import torch
import torch.multiprocessing
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss.location_loss import LocationCELoss
from .model.decoder import LocationDecoder
from .model.encoder import Encoder
from . import config as cfg
from .loss.get_loss import get_loss
from .model.net import LocationNet
from .utils.io import load_ckpt, save_ckpt
from .utils.netcdfloader import LocationNetCDFLoader, InfiniteSampler


def train(arg_file=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("* Number of GPUs: ", torch.cuda.device_count())

    cfg.set_train_args(arg_file)

    for subdir in ("", "/results", "/ckpt"):
        outdir = cfg.snapshot_dir + subdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    if not cfg.rotate_ensembles:
        torch.cuda.empty_cache()
        start_training_cycle(cfg.train_ensembles, cfg.val_ensembles)
    else:
        for i in range(cfg.resume_rotation,
                       min(cfg.resume_rotation + cfg.max_rotations, cfg.resume_rotation + len(cfg.train_ensembles))):
            val_ensembles = set(cfg.train_ensembles[i:i+1])
            train_ensembles = set(cfg.train_ensembles) - val_ensembles
            start_training_cycle(list(train_ensembles), list(val_ensembles), i)


def start_training_cycle(train_ensembles, val_ensembles, rotation=None):
    if rotation is not None:
        rotation_string = 'rotation_{}'.format(rotation)
    else:
        rotation_string = ''

    writer = SummaryWriter(log_dir='{}/{}'.format(cfg.log_dir, rotation_string))

    # create data sets
    dataset_train = LocationNetCDFLoader(cfg.data_root_dir, cfg.in_names, cfg.in_types, cfg.in_sizes, train_ensembles,
                                         cfg.train_ssis, cfg.locations, cfg.norm_to_ssi)
    dataset_val = LocationNetCDFLoader(cfg.data_root_dir, cfg.in_names, cfg.in_types, cfg.in_sizes, val_ensembles,
                                       cfg.val_ssis, cfg.locations, cfg.norm_to_ssi)

    iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=cfg.n_threads))

    encoder = Encoder
    decoder = LocationDecoder

    if cfg.mean_input:
        in_channels = len(cfg.in_names)
    else:
        in_channels = len(cfg.in_names) * cfg.time_steps

    model = LocationNet(encoder, decoder, img_size=cfg.in_sizes[0], in_channels=in_channels,
                        encoding_layers=cfg.encoding_layers, stride=(1, 1), bn=False).to(cfg.device)

    # define optimizer and loss functions
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    criterion = LocationCELoss()
    lambda_dict = cfg.LAMBDA_DICT_LOCATION
    if cfg.lr_scheduler_patience is not None:
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=cfg.lr_scheduler_patience)

    # define start point
    start_iter = 0
    if cfg.resume_iter:
        start_iter = load_ckpt(
            '{}/ckpt/{}{}.pth'.format(cfg.snapshot_dir, cfg.resume_iter, rotation_string), [('model', model)],
            cfg.device, [('optimizer', optimizer)])
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.lr
        print('Starting from iter ', start_iter)

    if cfg.multi_gpus:
        model = torch.nn.DataParallel(model)

    pbar = tqdm(range(start_iter, cfg.max_iter))
    for i in pbar:

        pbar.set_description("lr = {:.1e} {}".format(optimizer.param_groups[0]['lr'], rotation_string))

        # train model
        model.train()
        input, input_classes, _, _ = [x.to(cfg.device) for x in next(iterator_train)]
        output = model(input)
        train_loss = get_loss(criterion, lambda_dict, output, input_classes, writer, i, "train")

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if cfg.log_interval and (i + 1) % cfg.log_interval == 0:
            model.eval()
            data = []
            for j in range(2):
                data.append(torch.stack([dataset_val[k][j] for k in range(dataset_val.__len__())]))
            val_input, val_input_classes = data[0].to(cfg.device), data[1].to(cfg.device)

            with torch.no_grad():
                output = model(val_input)
            val_loss = get_loss(criterion, lambda_dict, output, val_input_classes, writer, i, "val")
            if cfg.lr_scheduler_patience is not None:
                lr_scheduler.step(val_loss)

        if (i + 1) % cfg.save_model_interval == 0 or (i + 1) == cfg.max_iter:
            save_ckpt('{:s}/ckpt/{:d}{}.pth'.format(cfg.snapshot_dir, i + 1, rotation_string),
                      [('model', model)], [('optimizer', optimizer)], i + 1)

    writer.close()


if __name__ == "__main__":
    train()
