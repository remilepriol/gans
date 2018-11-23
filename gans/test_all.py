import torch

from gans.utils import make_interpolation_noise, make_interpolation_samples
# Use this module with the pytest command line tool.

def test_make_interpolation_noise():
    interpolation, interpolated = \
        make_interpolation_noise(20, num_interpolation=10, num_interpolated=8)
    assert interpolation.size(1) == 20
    assert interpolation.size(0) == 10 * 8
    assert interpolated.size(0) == 8 * 2
    print('interpolation noise tested')


def test_make_interpolation_samples():
    samples = torch.randn([2 * 8, 3, 64, 64])
    interpolation = make_interpolation_samples(samples, num_interpolation=10)
    assert interpolation.size(0) == 8 * 10
    print('interpolation samples tested')
