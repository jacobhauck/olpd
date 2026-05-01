import mlx
import torch
import numpy as np
from operatorlearning.modules.basis import FullFourierBasis2d
from operatorlearning import GridFunction
import matplotlib.pyplot as plt


def pulse(x, y, sigma, config):
    r = ((x - config['xm'])**2 + (y - config['ym'])**2) ** 0.5
    mu = 6 * sigma
    uo = config['A'] * torch.exp(-(r - mu)**2 / (2 * sigma**2))
    vo = uo * (x - config['xm']) / (1e-8 + r)
    wo = uo * (y - config['ym']) / (1e-8 + r)
    return vo, wo


def error_curve(c, norm):
    return 1 - np.cumsum(sorted((c**2 / norm).cpu().numpy(), reverse=True))


def var_curve(basis, config):
    kx = basis.kx()
    ky = basis.ky()
    gx = kx / (2 * torch.pi)
    gy = ky / (2 * torch.pi)
    var = config['alpha'] / (config['beta'] + gx**2 + gy**2) ** config['gamma']
    return sorted((var / var.max()).cpu().numpy(), reverse=True)


@mlx.experiment
def run(config, name, group=None):
    basis = FullFourierBasis2d(config['num_modes'], config['x_min'], config['x_max'])
    xy = GridFunction.uniform_x(
        torch.tensor(config['x_min']),
        torch.tensor(config['x_max']),
        config['resolution']
    )

    x, y = xy[..., 0], xy[..., 1]

    area = (config['x_max'][0] - config['x_min'][0]) * (config['x_max'][1] - config['x_min'][1])

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    for sigma in config['sigma']:
        vo, wo = pulse(x, y, sigma, config)
        cv = basis.coefficients(vo[None, ..., None], xy[None])[0]
        cw = basis.coefficients(wo[None, ..., None], xy[None])[0]

        ev = error_curve(cv, torch.mean(vo**2) * area)
        ew = error_curve(cw, torch.mean(wo**2) * area)
        axes[0].plot(ev, label=f'$\sigma={sigma}$')
        axes[1].plot(ew, label=f'$\sigma={sigma}$')

    axes[0].plot(var_curve(basis, config), label='var')
    axes[1].plot(var_curve(basis, config), label='var')
    axes[1].set_xlabel('# basis functions')
    axes[0].set_ylabel('% squared error')
    axes[1].set_ylabel('% squared error')

    axes[0].legend()
    axes[1].legend()
    plt.show()
