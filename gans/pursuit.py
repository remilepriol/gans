"""Implement a generative pursuit between two generators and a discriminator."""

import argparse
import datetime
import getpass
import json
import os
import random
import subprocess

import tensorboardX
import torch

import gans.spectral_norm
from gans import models, utils
from gans.arguments import str2bool


def main():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--imageSize', type=int, default=32,
                        help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=10,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32,
                        help='Size of the output image from the generator')
    parser.add_argument('--ndf', type=int, default=32,
                        help='Size of the input image to the discriminator')

    # Optimization
    parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1d', type=float, default=0.5, help='beta1 for discriminator adam.')
    parser.add_argument('--beta1g', type=float, default=0.5, help='beta1 for generator adam.')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam.')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--manual-seed', type=int, help='manual seed')

    # GAN architecture
    # only WGAN
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
    parser.add_argument('--timedirectory', type=str2bool, default='true',
                        help='If true, use time for the log directory, '
                             'else only use hyper-parameters.')
    parser.add_argument('--resume', type=str2bool, default='true',
                        help='Resume the previous training if it finds a model in log dir.')

    opt = parser.parse_args()
    print(opt)

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
    if opt.server == 'elisa':
        root_path = os.path.join('/data/milatmp1', getpass.getuser(), 'pursuit')
    elif opt.server == 'local':
        root_path = '../results/pursuit'

    strpenalty = '0' if opt.lanbda <= 0 else f'{opt.lanbda}{opt.penalty}'
    strspectral = '_SN' if opt.spectral_norm else ''
    strbatchnorm = '' if opt.batch_norm else '_noBN'
    folder_name = (
        f'wgan_gp={strpenalty}{strspectral}{strbatchnorm}'
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

    writer = tensorboardX.SummaryWriter(opt.outf)

    # DEVICE
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('running on gpu')
    else:
        device = torch.device('cpu')
        print('running on cpu')

    # INITIALIZE MODELS
    netG0 = models.GeneratorNet(opt).to(device)
    netG1 = models.GeneratorNet(opt).to(device)
    netD = models.DiscriminatorNet(opt).to(device)

    print(netD)
    print(netG0)

    # input size of the discriminator
    sample_size = [opt.batchSize, 3, opt.imageSize, opt.imageSize]
    # input size of the generator
    latent_size = [opt.batchSize, opt.nz, 1, 1]
    # input noise to plot samples
    fixed_noise = torch.randn(latent_size, device=device)
    # labels
    labels0 = torch.zeros(opt.batchSize, device=device)
    labels1 = torch.ones(opt.batchSize, device=device)

    print('Setup optimizer')
    optimizerD = torch.optim.Adam(
        netD.parameters(), lr=opt.lr, betas=(opt.beta1d, opt.beta2))
    optimizerG = torch.optim.Adam(
        list(netG0.parameters()) + list(netG1.parameters()),
        lr=opt.lr, betas=(opt.beta1g, opt.beta2))

    for step in range(10 * opt.niter + 1):
        ############################
        # (1) Update D network: maximize D(G1(z)) - D(G0(z))
        ###########################

        # TODO: try using the same noise or different noise inputs
        latent_noise = torch.randn(latent_size, device=device)
        # latent_noise2 = torch.randn(latent_size, device=device)

        samples0 = netG0(latent_noise)
        samples1 = netG1(latent_noise)

        netD.zero_grad()
        pred0 = netD(samples0.detach()).squeeze()
        pred1 = netD(samples1.detach()).squeeze()

        lossD = - (pred1 - pred0).mean()
        lossD.backward()

        # gradient penalty
        if opt.lanbda > 0:
            inp = torch.cat([samples0, samples1], dim=0)
            gp = netD.gradient_penalty(inp.detach())
            (opt.lanbda * gp).backward()

        # update
        optimizerD.step()

        ############################
        # (2) Update both G networks: minimize D(G1(z)) - D(G0(z))
        ###########################
        netG0.zero_grad()
        netG1.zero_grad()

        pred0 = netD(samples0).squeeze()
        pred1 = netD(samples1).squeeze()

        lossG = (pred1 - pred0).mean()

        lossG.backward()
        optimizerG.step()

        # monitor
        accuracy = .5 * (torch.sigmoid(pred1).round().mean()
                         + 1 - torch.sigmoid(pred0).round().mean())

        if step % 10 == 0:  # print and log info
            print(f'step {step:<8}\t accuracy {accuracy:<6.2f}\t logit difference {lossG:<6.2f}')

            info = {
                'logit_dist': lossG,
                'accuracy': accuracy,
                'discriminator_gradient_norm': 0 if opt.lanbda <= 0 else gp.item()
            }
            info.update({f'sensitivity/{layer}': sensitivity
                         for layer, sensitivity in netD.get_sensitivity().items()})

            for tag, val in info.items():
                writer.add_scalar(tag, val, global_step=step)
                utils.tag2file(opt.outf, tag, [val], step)

        if step % 100 == 0:  # record weights spectrum for each layer
            layer_spectrum = gans.spectral_norm.spectrum(netD)

            for layer_id, spectrum in enumerate(layer_spectrum):
                writer.add_scalar(f'norms/spectral_layer{layer_id}',
                                  spectrum[0], global_step=step)
                utils.tag2file(opt.outf, f'spectrum/layer_{layer_id}', spectrum, step)

        if step % 100 == 0:  # plot samples
            utils.plot_images(
                opt.outf, writer, 'samples0', samples0, step)
            utils.plot_images(
                opt.outf, writer, 'samples1', samples1, step)

        # CHECKPOINT
        if step % 1000 == 0:
            checkpoint_file = f'{opt.outf}/state_{step}.pth'
            torch.save({
                'step': step,
                'generator0': netG0.state_dict(),
                'generator1': netG1.state_dict(),
                'discriminator': netD.state_dict(),
                'optimizer_generator': optimizerG.state_dict(),
                'optimizer_discriminator': optimizerD.state_dict()
            }, checkpoint_file)
