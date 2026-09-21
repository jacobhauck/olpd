import mlx
import matplotlib.pyplot as plt
import torch
import os
from math import log
from operatorlearning.data import OLDatasetLibrary, OLDataset
import set_fonts


@mlx.experiment
def make_plot(config, name, group=None):
    lib = OLDatasetLibrary('ad2d')
    datasets = [
        OLDataset(lib.dataset_path(config['split'], dataset_id, config.get('resolution')))
        for dataset_id in config['datasets']
    ]
    fig_size = ((1 + len(datasets)) * config['fig_size'] + 1, config['fig_size'] + 1)
    fig, axes = plt.subplots(1, 1 + len(datasets), figsize=fig_size)

    u, _, v, _ = datasets[0][config['index']]
    max_val = float(u.abs().max())

    im_kwargs = {
        'origin': 'lower',
        'vmin': -max_val,
        'vmax': max_val,
        'cmap': 'seismic'
    }

    var_dir = mlx.results_dir('total_variance', 'ad2d')
    var_data = torch.load(os.path.join(var_dir, f'{1}-train.pt'))
    max_dim = len(var_data['u_vals'])
    if 'u_vals_true' in var_data:
        var_data['u_vals_true'] = var_data['u_vals_true'][var_data['u_vals_true'] > 0]
        max_dim = len(var_data['u_vals_true'])

    u_vals = var_data['u_vals_true'][:max_dim]

    u_var = u_vals.sum()
    u_ent = max_dim * log(2 * torch.pi * torch.e) / 2 + torch.log(u_vals).sum() / 2

    axes[0].imshow(u[:, :, 0].T, **im_kwargs)
    axes[0].set_axis_off()
    axes[0].set_title(f'Initial condition\n$V={u_var.item():.03g}$\n$H={round(u_ent.item()/10)*10:g}$')

    for i, dataset in enumerate(datasets):
        _, _, v, _ = dataset[config['index']]
        var_data = torch.load(os.path.join(var_dir, f'{i+1}-train.pt'))
        max_dim = len(var_data['u_vals'])
        if 'u_vals_true' in var_data:
            var_data['u_vals_true'] = var_data['u_vals_true'][var_data['u_vals_true'] > 0]
            max_dim = len(var_data['u_vals_true'])

        v_vals = var_data['v_vals'][:max_dim]

        v_var = v_vals.sum()
        v_ent = max_dim * log(2*torch.pi*torch.e) / 2 + torch.log(v_vals).sum() / 2

        axes[i + 1].imshow(v[:, :, 0].T, **im_kwargs)
        axes[i + 1].set_axis_off()
        axes[i + 1].set_title(f'$k = {int(round(lib[config["datasets"][i]]["k"] / 1e-5)):d}k_0$\n$V={v_var.item():.03g}$\n$H={round(v_ent.item()/10)*10:g}$')

    fig.tight_layout()

    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bbox0 = get_bbox(axes[0])
    bbox1 = get_bbox(axes[1])
    x_middle = (bbox0.x1 + bbox1.x0) / 2

    line = plt.Line2D(
        [x_middle, x_middle], [0.1, 0.9],
        transform=fig.transFigure, color='black', linestyle='--'
    )
    fig.add_artist(line)

    if config.get('show', False):
        plt.show()

    mlx.show_and_save(fig, 'sample_solutions', config, name)
