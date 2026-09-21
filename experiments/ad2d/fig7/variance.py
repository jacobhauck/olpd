import mlx
import os
import torch
import matplotlib.pyplot as plt
from operatorlearning.data import OLDatasetLibrary
import set_fonts


@mlx.experiment
def plot(config, name, group=None):
    fig, ax = plt.subplots()

    var_dir = mlx.results_dir('total_variance', 'ad2d')
    lib = OLDatasetLibrary('ad2d')
    k = []
    variance = []
    for i in range(1, 6):
        var = torch.load(os.path.join(var_dir, f'{i}-train.pt'))['v_vals'].sum().item()
        k.append(round(lib[i]['k'] / lib[1]['k']))
        variance.append(var)

    ax.set_ylabel('Total Variance')
    ax.set_xlabel('$k / k_0$')
    ax.plot(k, variance)

    mlx.show_and_save(fig, 'cost-error-variance', config, name)
