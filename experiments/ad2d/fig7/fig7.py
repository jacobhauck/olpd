import mlx
import os
import json
import matplotlib.pyplot as plt
import set_fonts


@mlx.experiment
def plot_figure(config, name, group=None):
    with open(os.path.join(mlx.results_dir(name), 'test_error', 'error.json'), 'r') as f:
        error = json.load(f)

    fig, ax = plt.subplots()
    ax.set_xlabel('\\# basis functions ($p$)')
    ax.set_ylabel('Error (relative $L^2$)')
    for k, ps in error['ps'].items():
        errors = error['errors'][k]
        order = sorted(range(len(ps)), key=lambda i: ps[i])
        ps = [ps[i] for i in order]
        errors = [errors[i] for i in order]
        ax.plot(ps, errors, label=f'$k={k}k_0$')

    ax.legend()

    mlx.show_and_save(fig, 'plot', config, name)
