import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def l2normalize(v, eps=1e-8):
    return v / (v.norm() + eps)


def spectral_power_iteration(weight, u, height, n_power_iterations):
    assert n_power_iterations > 0
    matw = weight.data.view(height, -1)
    for _ in range(n_power_iterations):
        v = l2normalize(torch.mv(matw.t(), u))
        u = l2normalize(torch.mv(matw, v))

    # backprop through the last step of the calculation of sigma.
    # but not through u and v
    uvar = Variable(u, requires_grad=False)
    vvar = Variable(v, requires_grad=False)
    sigma = torch.sum(uvar * torch.mv(weight.view(height, -1), vvar))
    return sigma, u


class SNConv2d(nn.Conv2d):
    def __init__(self, *args, n_power_iterations=1, **kwargs):
        super(SNConv2d, self).__init__(*args, **kwargs)
        self.n_power_iterations = n_power_iterations
        self.height = self.weight.data.shape[0]
        self.register_buffer('u', l2normalize(self.weight.data.new(self.height).normal_(0, 1)))

    def forward(self, input):
        sigma, self.u = spectral_power_iteration(self.weight, self.u, self.height,
                                                 self.n_power_iterations)
        return F.conv2d(input, self.weight / sigma, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear):
    def __init__(self, *args, n_power_iterations=1, **kwargs):
        super(SNLinear, self).__init__(*args, **kwargs)
        self.n_power_iterations = n_power_iterations
        self.height = self.weight.data.shape[0]
        self.register_buffer('u', l2normalize(self.weight.data.new(self.height).normal_(0, 1)))

    def forward(self, input):
        sigma, self.u = spectral_power_iteration(self.weight, self.u, self.height,
                                                 self.n_power_iterations)
        return F.linear(input, self.weight / sigma, self.bias)


def spectrum(net):
    norms = []
    for m in net.modules():
        classname = m.__class__.__name__
        if 'Conv' in classname or 'Linear' in classname:

            w = m.weight.data.view(m.weight.size(0), -1)
            u, s, v = torch.svd(w)

            if 'SN' in classname:  # spectral norm -> normalize
                estimate, _ = spectral_power_iteration(m.weight, m.u, m.height, 1)
                s = s / estimate.data

            norms.append(s)

    return norms
