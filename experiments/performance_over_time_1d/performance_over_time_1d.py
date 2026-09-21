import mlx
import os
import matplotlib.pyplot as plt
import torch
from math import log
from operatorlearning import OLDatasetLibrary

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

    fig, axes = plt.subplots(1, 2, figsize=config['figure_size'], layout='constrained')

    axes[0].set_title('Wave Error')
    axes[0].set_xlabel('$T$')
    axes[0].set_ylabel('Error (relative $L^2$)')
    wave_lines = []
    wave_labels = []

    for model in wave_model_rows:
        times = [float(wave_data[row][1]) for row in wave_model_rows[model]]
        errors = [float(wave_data[row][error_col]) for row in wave_model_rows[model]]
        order = sorted(enumerate(times), key=lambda pair: pair[1])
        line = axes[0].plot([times[o[0]] for o in order], [errors[o[0]] for o in order], label=model)
        wave_lines.append(line[0])
        wave_labels.append(line[0].get_label())

    wave_vars = []
    wave_times = []
    wave_lib = OLDatasetLibrary('wave1d')
    wave_var_dir = mlx.results_dir('total_variance', 'wave1d')
    for i in range(1, 6):
        wave_times.append(wave_lib[i]['t_final'])
        var_data = torch.load(os.path.join(wave_var_dir, f'{i}-train.pt'))

        max_dim = len(var_data['u_vals'])
        if 'u_vals_true' in var_data:
            var_data['u_vals_true'] = var_data['u_vals_true'][var_data['u_vals_true'] > 0]
            max_dim = len(var_data['u_vals_true'])

        v_vals = var_data['v_vals_true'][:max_dim]
        v_ent = max_dim * log(2 * torch.pi * torch.e) / 2 + torch.log(v_vals).sum() / 2
        wave_vars.append(v_ent.item())

    wave_var_ax = axes[0].twinx()
    line = wave_var_ax.plot(wave_times, wave_vars, c='black', linestyle='--', label='$H$')
    wave_lines.append(line[0])
    wave_labels.append(line[0].get_label())
    wave_var_ax.set_ylim(min(wave_vars) * 2, 0)

    #axes[0].legend(wave_lines, wave_labels)

    heat_lines = []
    heat_labels = []

    axes[1].set_title('Heat Error')
    axes[1].set_xlabel('$T$')
    for model in heat_model_rows:
        times = [float(wave_data[row][1]) for row in heat_model_rows[model]]
        errors = [float(heat_data[row][error_col]) for row in heat_model_rows[model]]
        order = sorted(enumerate(times), key=lambda pair: pair[1])
        line = axes[1].plot([times[o[0]] for o in order], [errors[o[0]] for o in order], label=model)
        heat_lines.append(line[0])
        heat_labels.append(line[0].get_label())

    heat_vars = []
    heat_times = []
    heat_lib = OLDatasetLibrary('heat1d')
    heat_var_dir = mlx.results_dir('total_variance', 'heat1d')
    for i in range(1, 6):
        heat_times.append(heat_lib[i]['t_final'])
        var_data = torch.load(os.path.join(heat_var_dir, f'{i}-train.pt'))

        max_dim = len(var_data['u_vals'])
        if 'u_vals_true' in var_data:
            var_data['u_vals_true'] = var_data['u_vals_true'][var_data['u_vals_true'] > 0]
            max_dim = len(var_data['u_vals_true'])

        v_vals = var_data['v_vals_true'][:max_dim]
        v_ent = max_dim * log(2 * torch.pi * torch.e) / 2 + torch.log(v_vals).sum() / 2
        heat_vars.append(v_ent.item())
    heat_var_ax = axes[1].twinx()
    line = heat_var_ax.plot(heat_times, heat_vars, c='black', linestyle='--', label='$H$')
    heat_lines.append(line[0])
    heat_labels.append(line[0].get_label())
    heat_var_ax.set_ylabel('Entropy')
    heat_var_ax.set_ylim(min(heat_vars) * 1.1, 0)

    axes[1].legend(heat_lines, heat_labels)

    if config['show']:
        plt.show()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'compare-{config["split"]}.{config.get("format", "png")}')
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure at {output_file}')
