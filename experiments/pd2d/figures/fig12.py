import mlx
import matplotlib.pyplot as plt
import json
import os


@mlx.experiment
def plot(config, name, group=None):
    fig, ax = plt.subplots(2, 4, figsize=config['figure_size'])

    with open(os.path.join(mlx.results_dir('pd2d/figures/fig12_test_error'), 'error.json'), 'r') as f:
        error = json.load(f)

    for gamma in error:
        errors = error[gamma]['error']
        ps = error[gamma]['p']
        sort_order = sorted(range(len(errors)), key=lambda i: ps[i])
        errors = [errors[i] for i in sort_order]
        ps = [ps[i] for i in sort_order]
        ax.plot(ps, errors, label=f'$\gamma = {gamma:.1f}$')

    ax.set_xlabel('Model size (# basis functions)')
    ax.set_ylabel('Error (relative $L^2$)')
    ax.legend()

    mlx.show_and_save(fig, 'cost-error', config, name)
