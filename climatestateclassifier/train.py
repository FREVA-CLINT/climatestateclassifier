import os

import torch
import torch.multiprocessing
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as cfg
from .model.net import ClassificationNet
from .utils.io import load_ckpt, save_ckpt
from .utils.netcdfloader import NetCDFLoader, InfiniteSampler


def train(arg_file=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("* Number of GPUs: ", torch.cuda.device_count())

    cfg.set_train_args(arg_file)

    if not os.path.exists(cfg.snapshot_dir):
        os.makedirs(cfg.snapshot_dir)
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    if not cfg.rotate_samples:
        torch.cuda.empty_cache()
        start_training_cycle(cfg.train_samples, cfg.val_samples)
    else:
        for i in range(cfg.resume_rotation,
                       min(cfg.resume_rotation + cfg.max_rotations, cfg.resume_rotation + len(cfg.train_samples))):
            val_samples = set(cfg.train_samples[i:i+1])
            train_samples = set(cfg.train_samples) - val_samples
            start_training_cycle(list(train_samples), list(val_samples), i)


def start_training_cycle(train_samples, val_samples, rotation=None):
    if rotation is not None:
        rotation_string = 'rotation_{}'.format(rotation)
    else:
        rotation_string = ''

    writer = SummaryWriter(log_dir='{}/{}'.format(cfg.log_dir, rotation_string))

    # create data sets
    dataset_train = NetCDFLoader(cfg.data_files_train, cfg.data_types, cfg.time_steps)
    dataset_val = NetCDFLoader(cfg.data_files_val, cfg.data_types, cfg.time_steps)

    iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=cfg.n_threads))
    
    iterator_val = DataLoader(dataset_val, batch_size=cfg.batch_size,
                                     num_workers=cfg.n_threads)

    in_channels = len(cfg.data_types) * cfg.time_steps[0]

    if cfg.task=='prediction':
        n_output = 1
        criterion = nn.MSELoss()
    else:
        n_output = cfg.labels
        criterion = nn.CrossEntropyLoss()

    model = ClassificationNet(img_sizes=dataset_train.img_sizes[0],
                              in_channels=in_channels,
                              enc_dims=[dim for dim in cfg.encoder_dims],
                              dec_dims=[dim for dim in cfg.decoder_dims],
                              n_classes=n_output).to(cfg.device)

    # define optimizer and loss functions
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    

    # define start point
    start_iter = 0
    if cfg.resume_iter:
        start_iter = load_ckpt(
            '{}/{}{}.pth'.format(cfg.snapshot_dir, cfg.resume_iter, rotation_string), [('model', model)],
            cfg.device, [('optimizer', optimizer)])
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.lr
        print('Starting from iter ', start_iter)

    pbar = tqdm(range(start_iter, cfg.max_iter))
    for i in pbar:

        pbar.set_description("lr = {:.1e} {}".format(optimizer.param_groups[0]['lr'], rotation_string))

        # train model
        model.train()
        input, target = next(iterator_train)
        output = model(input.to(cfg.device))
        train_loss = criterion(output.squeeze(), target.to(cfg.device).float().squeeze())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss = train_loss.item()

        if cfg.log_interval and (i + 1) % cfg.log_interval == 0:
            model.eval()
            val_losses = []
            val_predictions = []
            for batch in iterator_val:
                input, target = batch
                input, target = input.to(cfg.device), target.to(cfg.device)
            
                with torch.no_grad():
                    output = model(input)

                val_loss = criterion(output.squeeze(), target.squeeze())
                val_losses.append(val_loss.item())
                val_predictions += list(output.numpy())
            val_loss = torch.tensor(val_losses).mean()
            
            writer.add_scalar("train-loss", train_loss, i+1)
            writer.add_scalar("val-loss", val_loss, i+1)

            if cfg.print_val_output:
                val_output_file = os.path.join(cfg.log_dir,f'val_predictions_iter_{i}.csv')
                np_out = np.array(val_predictions)
                np.savetxt(val_output_file, np_out)

        if (i + 1) % cfg.save_model_interval == 0 or (i + 1) == cfg.max_iter:
            save_ckpt('{:s}/{:d}{:s}.pth'.format(cfg.snapshot_dir, i + 1, rotation_string),
                      [('model', model)], [('optimizer', optimizer)], i + 1)

    writer.close()


if __name__ == "__main__":
    train()
