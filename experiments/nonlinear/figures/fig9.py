import mlx
import torch
import os
import matplotlib.pyplot as plt
from operatorlearning.data import OLDatasetLibrary


@mlx.experiment
def plot_var(config, name, group=None):
    mlx.configure_plotting(config)
    fig, axes = plt.subplots()

    lib = OLDatasetLibrary('nonlinear')

    for dataset_id in config['dataset_ids']:
        data_file = f'{dataset_id}-{config["split"]}.pt'
        data = torch.load(os.path.join(mlx.results_dir('total_variance', 'nonlinear'), data_file))

        max_dim = len(data['u_vals'])
        if 'max_dim' in config:
            max_dim = config['max_dim']

        if 'u_vals_true' in data:
            data['u_vals_true'] = data['u_vals_true'][data['u_vals_true'] > 0]
            max_dim = len(data['u_vals_true'])

        u_vals = data['u_vals'][:max_dim]
        v_vals = data['v_vals'][:max_dim]

        if dataset_id == config['dataset_ids'][0]:
            axes.plot(torch.sort(u_vals, descending=True).values, label='$u$ (input)', linestyle='--', color='black')

        axes.plot(torch.sort(v_vals, descending=True).values, label=f'$v$, $a = {lib[dataset_id]["a"]}$')

    axes.set_xlabel('Eigenvalue ordinal')
    axes.set_ylabel('Variance')
    if config.get('scale', 'log') != 'linear':
        axes.set_yscale('log')

    axes.legend()

    mlx.show_and_save(fig, 'total_variance', config, name)
