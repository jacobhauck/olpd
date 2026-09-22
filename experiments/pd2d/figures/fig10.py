import mlx
import torch
import os
import matplotlib.pyplot as plt
from operatorlearning.data import OLDatasetLibrary
import set_fonts


@mlx.experiment
def plot(config, name, group=None):
    fig, ax = plt.subplots(figsize=config['figure_size'])

    ax.set_xlabel('Eigenvalue ordinal')
    ax.set_ylabel('Variance')

    lib = OLDatasetLibrary('elastic2d')
    for i in range(1, 5):
        meta = lib[i]
        data = torch.load(os.path.join('results', 'total_variance', 'elastic2d', f'{i}-train.pt'))
        ax.plot(torch.sort(data['u_vals_true'], descending=True).values[:config['max_dim']], label=f'$\gamma = {meta["gamma"]:.1f}$')
    ax.legend()
    ax.set_yscale('log')

    mlx.show_and_save(fig, 'eigenvalues', config, name)
