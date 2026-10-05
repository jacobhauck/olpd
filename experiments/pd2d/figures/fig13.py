import mlx
import matplotlib.pyplot as plt
import json
import os


@mlx.experiment
def plot(config, name):
    mlx.configure_plotting(config)
    fig, ax = plt.subplots(figsize=config['figure_size'])

    with open(os.path.join(mlx.results_dir('pd2d/figures/fig13_test_error'), 'error.json'), 'r') as f:
        error = json.load(f)

    for gamma in error:
        errors = error[gamma]['error']
        ps = error[gamma]['p']
        sort_order = sorted(range(len(errors)), key=lambda i: ps[i])
        errors = [errors[i] for i in sort_order]
        ps = [ps[i] for i in sort_order]
        ax.plot(ps, errors, label=f'$\gamma = {gamma}$')

    ax.set_xlabel('Model size (\\# basis functions)')
    ax.set_ylabel('Error (relative $L^2$)')
    ax.legend()

    mlx.show_and_save(fig, 'cost-error', config, name)
