import torch
import torch.nn as nn
from torch.autograd import Variable

from gans.spectral_norm import SNConv2d


def upblock(in_channel, out_channel, mode, bias, batch_norm,
            stride=2, padding=1, last_layer=False):
    """Convolution block that up-scales the input by a factor 2."""

    if mode == 'convtranspose':
        block = [nn.ConvTranspose2d(in_channel, out_channel, 4,
                                    stride=stride, padding=padding, bias=bias)]
    else:
        block = [nn.Upsample(scale_factor=2, mode=mode),
                 nn.Conv2d(in_channel, out_channel, 5, stride=1, padding=2, bias=bias)]

    if batch_norm:
        block += [nn.BatchNorm2d(out_channel)]

    if last_layer:
        block += [nn.Tanh()]
    else:
        block += [nn.ReLU(True)]

    return block


class GeneratorNet(nn.Module):
    """Generator network with square layers of  size [1, 4, 8, 16, 32]"""

    def __init__(self, opt):
        super(GeneratorNet, self).__init__()
        self.ngpu = opt.ngpu

        # bias heuristic to reproduce negative momentum results
        batch_norm = True  # opt.batch_norm
        bias = False  # not opt.batch_norm

        channels = [opt.ngf * 8, opt.ngf * 4, opt.ngf * 2, opt.ngf]
        if opt.ngf == 32:  # remove the first layer
            channels = channels[1:]

        # input is Z, going into a convolution.
        # NO need for upsampling here as it starts from dimension 1.
        layers = upblock(opt.nz, channels[0], 'convtranspose',
                         bias, batch_norm, stride=1, padding=0)

        for i in range(len(channels) - 1):
            layers += upblock(channels[i], channels[i + 1], opt.upsample, bias, batch_norm)

        layers += upblock(channels[-1], opt.nc, opt.upsample, bias, batch_norm, last_layer=True)

        self.main = nn.Sequential(*layers)

        self.apply(weights_init)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class DownBlock(nn.Module):
    """Convolution block that downscales the input by a factor 2."""

    def __init__(self, in_channel, out_channel, spectral_norm, leakage, bias, batch_norm):
        super(DownBlock, self).__init__()

        conv2d = SNConv2d if spectral_norm else nn.Conv2d
        self.conv = conv2d(in_channel, out_channel, 4, stride=2, padding=1, bias=bias)

        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channel)
        else:
            self.batch_norm = None

        self.nonlinearity = nn.LeakyReLU(leakage, inplace=True)

    def forward(self, inp):
        out = self.conv(inp)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        out = self.nonlinearity(out)
        return out


class DiscriminatorNet(nn.Module):

    def __init__(self, opt):
        super(DiscriminatorNet, self).__init__()

        conv2d = SNConv2d if opt.spectral_norm else nn.Conv2d

        # bias heuristic to reproduce negative momentum results
        batch_norm = opt.batch_norm and not opt.spectral_norm
        bias = not batch_norm
        leakage = 1e-2 if opt.batch_norm else 0.2

        channels = [opt.ndf, opt.ndf * 2, opt.ndf * 4, opt.ndf * 8]
        if opt.ndf == 32:  # remove the last layer
            channels = channels[:-1]

        # input is (nc) x ndf x ndf (where ndf=64 or 32)
        self.layers = [DownBlock(opt.nc, channels[0], opt.spectral_norm,
                                 leakage, bias, batch_norm=False)]

        for i in range(len(channels) - 1):
            self.layers += [DownBlock(channels[i], channels[i + 1], opt.spectral_norm,
                                      leakage, bias, batch_norm)]

        # Linear layer with scalar output defined as a convolution for convenience
        self.layers += [conv2d(channels[-1], 1, 4, stride=1, padding=0, bias=False)]

        # never applied but necessary for the module to know its childens
        self.main = nn.Sequential(*self.layers)

        self.gradients = {}
        self.activations = {}

        self.apply(weights_init)

    def forward(self, inp):
        h = inp
        for depth, layer in enumerate(self.layers):
            h = layer(h)
            name = f'h{depth+1}'
            h.register_hook(self.save_grad(name))
            self.activations[name] = h

        return h.view(-1, 1).squeeze(1)

    def save_grad(self, name):
        """Save the gradient of a hidden unit in self.gradients dictionary."""

        def hook(grad):
            self.gradients[name] = grad

        return hook

    def get_sensitivity(self):
        """Return the mean <loss gradient, hidden> for each layer. """
        sensitivity = {}
        for k in self.gradients.keys():
            sensitivity[k] = (self.activations[k] * self.gradients[k]).mean(0).sum()

        return sensitivity

    def saliency(self, inp, requires_grad):
        inp = Variable(inp, requires_grad=True)
        o = self.forward(inp)
        return torch.autograd.grad(
            o, inp,
            grad_outputs=torch.ones_like(o),
            create_graph=requires_grad,
            only_inputs=requires_grad  # don't accumulate other gradients in .grad
        )[0], o

    def gradient_penalty(self, inp, requires_grad=True):
        g, _ = self.saliency(inp, requires_grad)
        gp = ((g.view(inp.size(0), -1).norm(p=2, dim=1)) ** 2).mean()
        return gp

    def wgan_gp(self, inp):
        g, _ = self.saliency(inp, requires_grad=True)
        gp = torch.mean(((g.view(inp.size(0), -1).norm(p=2, dim=1)) - 1) ** 2)
        return gp

    def fischer_gp(self, inp):
        g, out = self.saliency(inp, requires_grad=True)
        gp = torch.mean(
            out * (1 - out) *
            (g.view(inp.size(0), -1).norm(p=2, dim=1)) ** 2)
        return gp

    def gradient_penalty_g(self, z, netG):
        # inp = Variable(inp, requires_grad=True)
        o = self.forward(netG(z))
        g = torch.autograd.grad(
            o, netG.parameters(),
            grad_outputs=torch.ones_like(o),
            create_graph=True,
            only_inputs=True  # don't accumulate other gradients in .grad
        )[0]
        gp = 0
        for grad in g:
            gp += ((grad.view(z.size(0), -1).norm(p=2, dim=1)) ** 2).mean()
        return gp


def weights_init(m):
    """Custom weights initialization."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
