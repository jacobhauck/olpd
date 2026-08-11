import mlx
import os
import torch
import matplotlib.pyplot as plt
import set_fonts
from operatorlearning.data import OLDatasetLibrary


@mlx.experiment
def make_plots(config, name, group=None):
    output_dir = os.path.join('results', name)

    fig, axes = plt.subplots(1, 2, figsize=config['figure_size'], sharey=True)

    wave1d = OLDatasetLibrary('wave1d')
    heat1d = OLDatasetLibrary('heat1d')

    axes[0].set_title(f'Top-{config["num_vars"]} Wave Variances')
    axes[0].set_xlabel('$T$')
    axes[0].set_ylabel('Relative Variance')
    times = []
    out_var = []
    for k in range(config['num_vars']):
        out_var.append([])

    for heat_id in config['wave_datasets']:
        params = wave1d[heat_id]
        n = torch.arange(0, params['num_modes'] + 1).to(torch.float)  # (num_modes + 1,)
        size = params['x1'] - params['x0']
        lam_u = params['alpha'] / (params['beta'] + n**2) ** (params['gamma'])
        print(lam_u[0:5])
        c = params['c']
        t = params['t_final']
        pi = torch.pi

        lam_v_1 = lam_u.clone()
        lam_v_1[0] *= t**2
        lam_v_1[1:] = lam_u[1:] * (torch.cos(2*n[1:]*pi*c*t/size) ** 2 + (size/(2*n[1:]*pi*c) * torch.sin(2*pi*n[1:]*c*t/size)) ** 2)
        lam_v_2 = lam_u * (torch.cos(2*n*pi*c*t/size) ** 2 + ((2*n*pi*c)/size*torch.sin(2*pi*n*c*t/size)) ** 2)
        lam_v = torch.sort(torch.cat([lam_v_1, lam_v_2]), descending=True).values[config['exclude']:]

        for k in range(config['num_vars']):
            out_var[k].append(lam_v[k].item())

        times.append(t)

    print(times)
    print(out_var[0])
    for k in range(config['num_vars']):
        order = sorted(enumerate(times), key=lambda pair: pair[1])
        axes[0].plot([times[o[0]] for o in order], [out_var[k][o[0]] / out_var[k][order[0][0]] for o in order])

    axes[1].set_title(f'Top-{config["num_vars"]} Heat Variances')
    axes[1].set_xlabel('$T$')
    times = []
    out_var = []
    for k in range(config['num_vars']):
        out_var.append([])

    for heat_id in config['heat_datasets']:
        params = heat1d[heat_id]
        n = torch.arange(0, params['num_modes'] + 1).to(torch.float)  # (num_modes + 1,)
        size = params['x1'] - params['x0']
        lam_u = params['alpha'] / (params['beta'] + n**2) ** (params['gamma'])
        k = params['k']
        t = params['t_final']
        pi = torch.pi

        lam_v = lam_u * torch.exp(-k * 4 * n**2 * pi**2 / size**2 * t)
        lam_v = torch.sort(lam_v, descending=True).values[config['exclude']:]

        for i in range(config['num_vars']):
            out_var[i].append(lam_v[i].item())

        times.append(t)
    print(times)
    print(out_var[0])
    for k in range(config['num_vars']):
        order = sorted(enumerate(times), key=lambda pair: pair[1])
        axes[1].plot([times[o[0]] for o in order], [out_var[k][o[0]] / out_var[k][order[0][0]] for o in order])

    if config['show']:
        plt.show()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'compare.{config.get("format", "png")}')
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure at {output_file}')
