import mlx
import os
import matplotlib.pyplot as plt
import set_fonts
import csv


@mlx.experiment
def make_plots(config, name, group=None):
    error_col = 4 if config['split'] == 'train' else 6

    output_dir = os.path.join('results', name)
    with open(config['wave_runs'], 'r', newline='') as f:
        wave_data = list(csv.reader(f))

    wave_model_rows = {}
    last = ''
    for i, row in enumerate(wave_data):
        if i == 0:
            continue
        if row[0] != '':
            wave_model_rows[row[0]] = []
            last = row[0]

        wave_model_rows[last].append(i)

    with open(config['heat_runs'], 'r', newline='') as f:
        heat_data = list(csv.reader(f))

    heat_model_rows = {}
    last = ''
    for i, row in enumerate(wave_data):
        if i == 0:
            continue
        if row[0] != '':
            heat_model_rows[row[0]] = []
            last = row[0]

        heat_model_rows[last].append(i)

    fig, axes = plt.subplots(1, 2, figsize=config['figure_size'])

    axes[0].set_title('Wave Error')
    axes[0].set_xlabel('$T$')
    axes[0].set_ylabel('Error (relative $L^2$)')

    for model in wave_model_rows:
        times = [float(wave_data[row][1]) for row in wave_model_rows[model]]
        errors = [float(wave_data[row][error_col]) for row in wave_model_rows[model]]
        order = sorted(enumerate(times), key=lambda pair: pair[1])
        axes[0].plot([times[o[0]] for o in order], [errors[o[0]] for o in order], label=model)

    axes[0].legend()

    axes[1].set_title('Heat Error')
    axes[1].set_xlabel('$T$')
    for model in heat_model_rows:
        times = [float(wave_data[row][1]) for row in heat_model_rows[model]]
        errors = [float(heat_data[row][error_col]) for row in heat_model_rows[model]]
        order = sorted(enumerate(times), key=lambda pair: pair[1])
        axes[1].plot([times[o[0]] for o in order], [errors[o[0]] for o in order], label=model)

    axes[1].legend()

    if config['show']:
        plt.show()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'compare-{config["split"]}.{config.get("format", "png")}')
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure at {output_file}')
