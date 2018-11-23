import os
import subprocess

import tensorboardX
import torch
import torch.nn as nn
import torchvision.datasets as vdset
import torchvision.transforms as vtransforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import gans.spectral_norm
from gans import arguments, models, scores, utils

sigmoid = nn.Sigmoid()

if __name__ == '__main__':

    opt = arguments.get_arguments()

    writer = tensorboardX.SummaryWriter(opt.outf)

    # DATA
    print(f'Loading dataset {opt.dataset} at {opt.datatmp}')
    transform = vtransforms.Compose([
        vtransforms.Resize(opt.imageSize),
        vtransforms.CenterCrop(opt.imageSize),
        vtransforms.ToTensor(),
        vtransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if opt.dataset == 'cifar10':  # torchvision dataset
        dataset = vdset.CIFAR10(opt.datatmp, download=True, transform=transform)

    else:  # folder dataset
        print(f'Copying dataset {opt.dataset} from {opt.dataroot} to {opt.datatmp}')
        if not os.path.isdir(opt.datatmp):
            os.makedirs(opt.datatmp, exist_ok=True)
        # rsync -ru stands for recusively updated
        subprocess.call(['rsync', '-ru', opt.dataroot, opt.datatmp])

        dataset = vdset.ImageFolder(root=opt.datatmp, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
        drop_last=True
    )
    print('Dataloader done')

    # INITIALIZE MODELS
    netG = models.GeneratorNet(opt)
    netD = models.DiscriminatorNet(opt)

    print(netD)
    print(netG)

    print('Define criterion and input tensors')
    criterion = nn.BCEWithLogitsLoss()
    if opt.mode == 'lsgan':
        criterion = nn.MSELoss()
    if opt.mode == 'wgan':
        def criterion(out, target):
            return ((1 - 2 * target) * out).mean()

    # input to the discriminator of size [opt.batchSize, 3, opt.imageSize, opt.imageSize]
    sample_noise = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    # input noise for the generator
    latent_noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
    # input noise to plot samples
    fixed_noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).normal_(0, 1)

    # labels to use with criterion
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    print('Move model and tensors to GPU')
    if opt.cuda:
        netD.cuda()
        netG.cuda()
        label = label.cuda()
        sample_noise = sample_noise.cuda()
        latent_noise = latent_noise.cuda()
        fixed_noise = fixed_noise.cuda()

    fixed_noise = Variable(fixed_noise)

    print('Setup optimizer')
    optimizerD = torch.optim.Adam(
        netD.parameters(), lr=opt.lr, betas=(opt.beta1d, opt.beta2))
    optimizerG = torch.optim.Adam(
        netG.parameters(), lr=opt.lr, betas=(opt.beta1g, opt.beta2))

    # optionally resume from a checkpoint
    opt.last_checkpoint = f'{opt.outf}/last_state.pth'
    opt.start_epoch = 1
    if opt.resume:
        if os.path.isfile(opt.last_checkpoint):
            print(f'=> loading checkpoint {opt.last_checkpoint}')
            checkpoint = torch.load(opt.last_checkpoint)
            opt.start_epoch = checkpoint['epoch']
            netG.load_state_dict(checkpoint['generator'])
            netD.load_state_dict(checkpoint['discriminator'])
            optimizerG.load_state_dict(checkpoint['optimizer_generator'])
            optimizerD.load_state_dict(checkpoint['optimizer_discriminator'])
            print(f'=> loaded checkpoint from epoch {opt.start_epoch}')

    print('Begin training')
    step = (opt.start_epoch - 1) * len(dataloader)
    for epoch in range(opt.start_epoch, opt.niter + 1):
        for i, data in enumerate(dataloader):
            step += 1
            for _ in range(opt.critic_iter):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                # train with real
                netD.zero_grad()

                real_cpu, _ = data
                batch_size = real_cpu.size(0)

                real_samples = real_cpu.cuda() if opt.cuda else real_cpu.clone()
                real = Variable(real_samples)
                label.resize_(batch_size).fill_(real_label)
                labelv = Variable(label)

                real_out = netD(real).squeeze()  # forward
                errD_real = criterion(real_out, labelv)
                errD_real.backward()

                real_sensitivity = netD.get_sensitivity()

                # train with fake
                latent_noise.resize_(batch_size, opt.nz, 1, 1).normal_(0, 1)
                noisev = Variable(latent_noise)
                labelv = Variable(label.fill_(fake_label))

                fake = netG(noisev)
                fake_out = netD(fake.detach()).squeeze()  # forward
                errD_fake = criterion(fake_out, labelv)
                errD_fake.backward()

                fake_sensitivity = netD.get_sensitivity()

                # gradient penalty
                gp = 0
                if opt.lanbda > 0:

                    if opt.penalty == 'grad_g':
                        # penalize gradient wrt generator's parameters
                        gp = netD.gradient_penalty_g(noisev.detach(), netG)

                    elif opt.penalty == 'wgangp':
                        # interpolate and subtract 1
                        intercoeffs = torch.FloatTensor(
                            batch_size, 1, 1, 1).uniform_(0, 1)
                        if opt.cuda:
                            intercoeffs = intercoeffs.cuda()
                        inp = real.data * intercoeffs + fake.data * (1 - intercoeffs)
                        gp = netD.wgan_gp(inp)
                    elif opt.penalty == 'fischer':
                        # penalise trace Fischer matrix of the discriminant
                        batch_half = int(batch_size / 2)
                        inp = torch.cat([
                            real.data[:batch_half],
                            fake.data[:batch_half]],
                            dim=0)
                        gp = netD.fischer_gp(inp)

                    else:
                        # penalize gradient wrt samples
                        if opt.penalty == 'real':
                            inp = real.data
                        elif opt.penalty == 'fake':
                            inp = fake.data
                        elif opt.penalty == 'both':
                            batch_half = int(batch_size / 2)
                            inp = torch.cat([
                                real.data[:batch_half],
                                fake.data[:batch_half]],
                                dim=0)
                        elif opt.penalty == 'uniform':
                            inp = sample_noise.uniform_(-1, 1)
                        elif opt.penalty == 'midinterpol':
                            inp = 0.5 * (real.data + fake.data)
                        gp = netD.gradient_penalty(inp)

                    (opt.lanbda * gp).backward()

                # update
                optimizerD.step()

                # monitor
                real_acc = sigmoid(real_out).data.round().mean()
                f_x = real_out.data.mean()
                D_x = sigmoid(real_out).data.mean()

                fake_acc = 1 - sigmoid(fake_out).data.round().mean()
                f_G_z1 = fake_out.data.mean()
                D_G_z1 = sigmoid(fake_out).data.mean()
                errD = errD_real + errD_fake

            for k in range(opt.gen_iter):
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                if k > 0:
                    latent_noise.resize_(opt.batchSize, opt.nz, 1, 1).normal_(0, 1)
                    noisev = Variable(latent_noise)
                    fake = netG(noisev)

                netG.zero_grad()

                fake_out = netD(fake).squeeze()
                if opt.mode == 'nsgan':  # non-saturating gan
                    # use the real labels (1) for generator cost
                    labelv = Variable(label.fill_(real_label))
                    errG = criterion(fake_out, labelv)
                elif opt.mode == 'mmgan':  # minimax gan
                    # use fake labels and opposite of criterion
                    labelv = Variable(label.fill_(fake_label))
                    errG = - criterion(fake_out, labelv)
                elif opt.mode == 'lsgan':  # least square gan NOT WORKING
                    # use real labels for generator
                    labelv = Variable(label.fill_(real_label))
                    errG = criterion(fake_out, labelv)
                elif opt.mode == 'wgan':
                    labelv = Variable(label.fill_(real_label))
                    errG = criterion(fake_out, labelv)

                errG.backward()
                optimizerG.step()

                # monitor
                f_G_z2 = fake_out.data.mean()
                D_G_z2 = sigmoid(fake_out).data.mean()

            if step % 100 == 0:  # print and log info
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f | %.4f'
                      % (epoch, opt.niter, i + 1, len(dataloader),
                         errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

                netD.get_sensitivity()

                # measure gradient norm on real and fake data
                real = data[0]
                if opt.cuda:
                    real = real.cuda()
                gp_real = netD.gradient_penalty(real, requires_grad=False)

                latent_noise.normal_(0, 1)
                fake = netG(Variable(latent_noise, requires_grad=False))
                gp_fake = netD.gradient_penalty(fake.data, requires_grad=False)

                info = {
                    'losses/disc_cost': errD.data[0],
                    'losses/gen_cost': errG.data[0],
                    'losses/logit_dist': f_x - f_G_z1,
                    'losses/penalty': 0 if opt.lanbda <= 0 else opt.lanbda * gp.data[0],
                    'values/f_x': f_x,
                    'values/f_G_z': f_G_z1,
                    'values/D_x': D_x,
                    'values/D_G_z': D_G_z1,
                    'values/grad_f_x': gp_real,
                    'values/grad_f_G_z': gp_fake,
                    'accuracy/real': real_acc,
                    'accuracy/fake': fake_acc,
                }

                for layer in fake_sensitivity.keys():
                    key = f'sensitivity/{layer}'
                    info[key] = (fake_sensitivity[layer] + real_sensitivity[layer]) / 2
                    # info['real_' + key] = real_sensitivity[layer]
                    # info['fake_' + key] = fake_sensitivity[layer]

                for tag, val in info.items():
                    writer.add_scalar(tag, val, global_step=step)
                    utils.tag2file(opt.outf, tag, [val], step)

            if step % 100 == 0:  # record weights spectrum for each layer
                layer_spectrum = gans.spectral_norm.spectrum(netD)

                for layer_id, spectrum in enumerate(layer_spectrum):
                    writer.add_scalar(f'norms/spectral_layer{layer_id}',
                                      spectrum[0], global_step=step)
                    utils.tag2file(opt.outf, f'spectrum/layer_{layer_id}', spectrum, step)

            if step % 500 == 0:  # plot samples
                utils.plot_images(
                    opt.outf, writer, 'real_samples', real_samples, step)
                fake = netG(fixed_noise)
                utils.plot_images(
                    opt.outf, writer, 'fake_samples', fake.data, step)
                interpolation_noise, interpolated_noise = \
                    utils.make_interpolation_noise(opt.nz)

                # # interpolation in the latent space
                # interpolation_noise = Variable(interpolation_noise).cuda()
                # fake_interpolation = netG(interpolation_noise)
                # utils.plot_images(
                #     opt.outf, writer, 'fake_interpolation_samples',
                #     fake_interpolation.data, step, nrow=10)

            if step % 1000 == 0:  # compute scores
                noise = torch.FloatTensor(10000, opt.nz, 1, 1).normal_(0, 1)
                if opt.cuda:
                    noise = noise.cuda()
                noise = Variable(noise, volatile=True)
                fake = netG(noise)

                inception, _ = scores.inception_score(fake.data, resize=True)
                # mode, _ = scores.mode_score(fake.data, real_samples, resize=True)

                # ishaan = scores.ishaan_inception(fake.data)

                print(f'Inception : {inception} and Mode score: 0\n')
                writer.add_scalar('metrics/inception_score', inception, global_step=step)
                # writer.add_scalar('metrics/inception10splits', inception10, global_step=step)
                # writer.add_scalar('metrics/mode_score', mode, global_step=step)

        # CHECKPOINT
        if epoch % 5 == 0:
            checkpoint_file = f'{opt.outf}/state_epoch={epoch}.pth'
            torch.save({
                'epoch': epoch,
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'optimizer_generator': optimizerG.state_dict(),
                'optimizer_discriminator': optimizerD.state_dict()
            }, checkpoint_file)
            subprocess.call(['cp', checkpoint_file, opt.last_checkpoint])
