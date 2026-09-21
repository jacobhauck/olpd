import mlx
import torch
import os
import matplotlib.pyplot as plt
from operatorlearning.data import OLDatasetLibrary, OLDataset
import set_fonts


@mlx.experiment
def plot(config, name, group=None):
    fig, axes = plt.subplots(1, 4, figsize=(4 * config['im_size'], config['im_size']))

    lib = OLDatasetLibrary('elastic2d')
    data = OLDataset(lib.dataset_path('test', 1))
    u, _, _, _ = data[config['sample_id']]
    im_kwargs = {
        'cmap': 'seismic',
        'origin': 'lower'
    }
    for i in range(1, 5):
        meta = lib[i]
        data = OLDataset(lib.dataset_path('test', i))
        u, _, _, _ = data[config['sample_id']]
        var_data = torch.load(os.path.join('results', 'total_variance', 'elastic2d', f'{i}-train.pt'))
        u_vals = var_data['u_vals_true'][var_data['u_vals_true'] > 0]
        entropy = torch.sum(torch.log(u_vals)).item()
        v_max = u.abs().max().item()
        im_kwargs['vmin'] = -v_max
        im_kwargs['vmax'] = v_max

        ax = axes[i-1]
        ax.set_axis_off()
        ax.set_title(f'$\gamma = {meta["gamma"]:.1f}$, $H = {entropy:.04g}$')
        ax.imshow(u[:, :, 0].T, **im_kwargs)

    mlx.show_and_save(fig, 'sample-initial', config, name)
