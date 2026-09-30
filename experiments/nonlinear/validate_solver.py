import mlx
import torch
import matplotlib.pyplot as plt
from .generate import solve_equation_explicit


def psi_true(x, y, t):
    return 0.1 * torch.cos(2 * torch.pi * x) * torch.sin(2 * torch.pi * (y - t))


@mlx.experiment
def validate(config, name, group=None):
    torch.set_default_dtype(torch.double)
    x = torch.linspace(config['x0'], config['x1'], config['mesh_size'] + 1)[:-1]
    dx = (x[1] - x[0])
    x += dx / 2
    y = torch.linspace(config['y0'], config['y1'], config['mesh_size'] + 1)[:-1]
    dy = (y[1] - y[0])
    y += dy / 2
    xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)

    def f(psi, x_, y_, t):
        psi_t = psi_true(x_, y_, t).to(psi.device)
        return (-4*torch.pi**2 + config['c']**2 * 8 * torch.pi ** 2) * psi_t + psi_t ** 3 - psi ** 3

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    u = psi_true(xy[..., 0], xy[..., 1], 0)
    im0 = axes[0].imshow(u, vmin=-.2, vmax=.2)
    im1 = axes[1].imshow(u, vmin=-.2, vmax=.2)
    im2 = axes[2].imshow((u - u).abs(), vmin=0, vmax=0.15)
    axes[0].set_title('true, 0, 0')
    axes[1].set_title('approx, 0, 0')
    axes[2].set_title('error')
    plt.pause(0.5)

    def callback(psi, n, t):
        if n % 10 == 0:
            psi_comp = psi_true(xy[..., 0], xy[..., 1], t)
            error = (psi.cpu() - psi_comp).abs().mean()
            im0.set_data(psi_true(xy[..., 0], xy[..., 1], t))
            im1.set_data(psi.cpu())
            im2.set_data((psi.cpu() - psi_comp).abs())
            axes[0].set_title(f'True n = {n}, t = {t:.02g}')
            axes[1].set_title(f'Approx error = {error.item():.02e}')
            plt.pause(0.01)

    v = -0.1*2*torch.pi * torch.cos(2*torch.pi*xy[..., 0]) * torch.cos(2*torch.pi*xy[..., 1])
    solve_equation_explicit(u, f, xy, dx, dy, config, velocity=v, callback=callback)
