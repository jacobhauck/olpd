import mlx
import torch
from scipy.sparse.linalg import splu
from .generate import (
    build_matrix,
    solve_equation,
    solve_equation_implicit1,
    solve_equation_implicit2
)
from math import ceil
import matplotlib.pyplot as plt


@mlx.experiment
def run_experiment(config, name, group=None):
    torch.set_default_dtype(torch.double)

    x = torch.linspace(config['x0'], config['x1'], config['mesh_size'] + 1)[:-1]
    dx = (x[1] - x[0])
    x += dx / 2

    y = torch.linspace(config['y0'], config['y1'], config['mesh_size'] + 1)[:-1]
    dy = (y[1] - y[0])
    y += dy / 2

    xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
    x_, y_ = xy[..., 0], xy[..., 1]
    # (m, m, 2)

    sx = torch.sin(2 * torch.pi * x_)
    cx = torch.cos(2 * torch.pi * x_)
    sy = torch.sin(2 * torch.pi * y_)
    cy = torch.cos(2 * torch.pi * y_)
    t = torch.tensor(float(config['t_final']))

    def f(t_):
        t_ = torch.tensor(t_)
        f_val = sx * cy * torch.cos(t_) / 2
        f_val += -config['c'][0] * 2 * torch.pi * cx * cy * (torch.sin(t_) / 2 + 1)
        f_val += config['c'][1] * 2 * torch.pi * sx * sy * (torch.sin(t_) / 2 + 1)
        f_val += 8 * config['k'] * torch.pi**2 * sx * cy * (1 + torch.sin(t_) / 2)
        if config['method'] == 'explicit':
            return f_val.to(config['device'])
        else:
            return f_val.numpy().reshape(-1)

    u = sx * cy
    v_true = u * (1 + torch.sin(t)/2)

    matrix1 = build_matrix(config['c'], config['k'], tuple(x_.shape), dx, dy, config['dt'])
    lu1 = splu(matrix1)
    matrix2 = build_matrix(config['c'], config['k'], tuple(x_.shape), dx, dy, 2/3*config['dt'])
    lu2 = splu(matrix2)
    time_steps = ceil(config['t_final'] / config['dt'])

    if config['method'] == 'explicit':
        v = solve_equation(u, config['c'], config['k'], dx, dy, config['dt'], time_steps, config['device'], f=f)
    elif config['method'] == 'implicit1':
        v = solve_equation_implicit1(u, lu1, config['dt'], time_steps, f=f)
    elif config['method'] == 'implicit2':
        v = solve_equation_implicit2(u, lu1, lu2, config['dt'], time_steps, f=f)
    else:
        raise ValueError('Invalid solver')

    print('max error:', (v - v_true).abs().max().item())
    print('l2 error:', ((v - v_true)**2).mean().item())
    fig, axes = plt.subplots(1, 2)
    args = {
        'origin': 'lower',
        'extent': (config['x0'], config['x1'], config['y0'], config['y1'])
    }
    axes[0].imshow(v_true.T, **args, vmin=-1.5, vmax=1.5)
    axes[0].set_title('True solution')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    axes[1].imshow(v.T, **args, vmin=-1.5, vmax=1.5)
    axes[1].set_title('Numerical solution')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$y$')
    plt.show()
