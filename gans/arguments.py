"""Arguments parser."""
import argparse
import datetime
import getpass
import json
import os
import random
import subprocess

import torch


def get_arguments():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', default='celebsmall')
    parser.add_argument('--imageSize', type=int, default=64,
                        help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64,
                        help='Size of the output image from the generator')
    parser.add_argument('--ndf', type=int, default=64,
                        help='Size of the input image to the discriminator')

    # Optimization
    parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1d', type=float, default=0.5, help='beta1 for discriminator adam.')
    parser.add_argument('--beta1g', type=float, default=0.5, help='beta1 for generator adam.')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam.')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--manual-seed', type=int, help='manual seed')

    # GAN architecture
    parser.add_argument('--mode', type=str, default='wgan',
                        help='Type of GAN: minimax, non-saturating, least-square, Wasserstein.',
                        choices=['mmgan', 'nsgan', 'lsgan', 'wgan'])
    parser.add_argument('--upsample', type=str, default='nearest',
                        choices=['convtranspose', 'nearest', 'bilinear'],
                        help='Method used in the generator to up-sample images.')
    parser.add_argument('--lanbda', type=float, default=0,
                        help='Regularization factor for the gradient penalty.')
    parser.add_argument('--penalty', type=str, default='both',
                        choices=['real', 'fake', 'both', 'uniform',
                                 'midinterpol', 'grad_g', 'wgangp', 'fischer'],
                        help='Distribution on which to apply gradient penalty.')
    parser.add_argument('--spectral-norm', type=str2bool, default='false',
                        help='If True, use spectral normalization in the discriminator.')
    parser.add_argument('--batch-norm', type=str2bool, default='true',
                        help='If True, use batch normalization in the discriminator.')

    # Checkpoints
    parser.add_argument('--server', default='local', choices=['local', 'elisa'])
    parser.add_argument('--timedirectory', type=str2bool, default='false',
                        help='If true, use time for the log directory, '
                             'else only use hyper-parameters.')
    parser.add_argument('--resume', type=str2bool, default='true',
                        help='Resume the previous training if it finds a model in log dir.')

    opt = parser.parse_args()
    print(opt)

    # DATASET
    opt.dataset = opt.dataset.lower()

    if opt.server == 'elisa':
        user = getpass.getuser()
        opt.datatmp = os.path.join('/Tmp', user, opt.dataset)

        if opt.dataset == 'celeba':
            opt.dataroot = '/data/lisa/data/celeba'
        elif opt.dataset == 'tinyimagenet':
            opt.dataroot = '/data/lisa/data/tiny-imagenet-200/train'
        elif opt.dataset == 'celebsmall':
            opt.dataroot = f'/data/milatmp1/{user}/celebsmall'

    if opt.server == 'local':
        if opt.dataset == 'celebsmall':
            opt.dataroot = '../celebsmall/'
            opt.datatmp = opt.dataroot

    # Number of input channels (RGB)
    opt.nc = 3

    # SEED
    if opt.manual_seed is None:
        opt.seed = random.randint(1, 10000)
    else:
        opt.seed = opt.manual_seed
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    # OUT FOLDER
    if opt.server=='elisa':
        root_path = os.path.join('/data/milatmp1', user, 'gans', opt.dataset)
    elif opt.server=='local':
        root_path = os.path.join('../results/', opt.dataset)

    strpenalty = '0' if opt.lanbda <= 0 else f'{opt.lanbda}{opt.penalty}'
    strspectral = '_SN' if opt.spectral_norm else ''
    strbatchnorm = '' if opt.batch_norm else '_noBN'
    folder_name = (
        f'{opt.mode}_gp={strpenalty}{strspectral}{strbatchnorm}'
        f'_lr={opt.lr}_beta1d={opt.beta1d}_beta1g={opt.beta1g}_beta2={opt.beta2}'
        f'_upsample={opt.upsample}'
    )
    if opt.timedirectory:  # new folder for each run
        now = datetime.datetime.now()
        folder_path = (
            f'{now.month}_{now.day}'
            f'/{now.hour}_{now.minute}_{folder_name}'
        )
    else:  # one folder per set of hyper-parameters
        folder_path = folder_name
    opt.outf = os.path.join(root_path, folder_path)
    print('Out folder: ', opt.outf)

    if not opt.resume:
        if os.path.isdir(opt.outf):
            subprocess.call('rm', '-r', opt.outf)

    os.makedirs(opt.outf, exist_ok=True)
    with open(os.path.join(opt.outf, 'config.json'), 'w') as fp:
        json.dump(vars(opt), fp, indent=4)

    return opt


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
