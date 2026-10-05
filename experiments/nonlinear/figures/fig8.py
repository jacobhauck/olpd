import os

import matplotlib.pyplot as plt
import mlx
import torch
from operatorlearning.data import OLDataset, OLDatasetLibrary


@mlx.experiment
def plot_figure(config, name):
    mlx.configure_plotting(config)
    fig, axes = plt.subplots(1, 2 + len(config['nonlinear_ids']), figsize=config['figure_size'])

    lib = OLDatasetLibrary('nonlinear')
    data_linear = OLDataset(lib.dataset_path(config['split'], config['linear_id']))
    u_l, _, v_l, _ = data_linear[config['sample']]

    im_kwargs = {
        'cmap': 'seismic',
        'origin': 'lower'
    }

    data = torch.load(os.path.join(mlx.results_dir('total_variance', 'nonlinear'), f'{config["linear_id"]}-train.pt'))
    input_var = data['u_vals'].sum().item()
    total_var = data['v_vals'].sum().item()

    max_val = u_l.abs().max().item()
    axes[0].imshow(u_l[:, :, 0].T, **im_kwargs, vmin=-max_val, vmax=max_val)
    axes[0].set_axis_off()
    axes[0].set_title(f'$u$ (initial)\n$V={input_var:.2f}$')

    max_val = v_l.abs().max().item()
    axes[1].imshow(v_l[:, :, 0].T, **im_kwargs, vmin=-max_val, vmax=max_val)
    axes[1].set_axis_off()
    axes[1].set_title(f'$v$ (\\textbf{{linear}}, $a = 0$)\n$V={total_var:.2f}$')

    for i in range(len(config['nonlinear_ids'])):
        dataset_id = config['nonlinear_ids'][i]
        data_nonlinear = OLDataset(lib.dataset_path(config['split'], dataset_id))
        data = torch.load(os.path.join(mlx.results_dir('total_variance', 'nonlinear'), f'{dataset_id}-train.pt'))
        total_var = data['v_vals'].sum().item()
        u_n, _, v_n, _ = data_nonlinear[config['sample']]
        max_val = v_n.abs().max().item()
        axes[i + 2].imshow(v_n[:, :, 0].T, **im_kwargs, vmin=-max_val, vmax=max_val)
        axes[i + 2].set_axis_off()
        axes[i + 2].set_title(f'$v$ ($a = {int(lib[dataset_id]["a"])}$)\n$V={total_var:.2f}$')

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

    mlx.show_and_save(fig, 'sample', config, name)
