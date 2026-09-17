import mlx
import torch
import matplotlib.pyplot as plt
from math import log
import set_fonts


@mlx.experiment
def plot_variance(config, name, group=None):
    fig, axes = plt.subplots()

    data = torch.load(config['var_data_file'])

    max_dim = len(data['u_vals'])
    if 'u_vals_true' in data:
        data['u_vals_true'] = data['u_vals_true'][data['u_vals_true'] > 0]
        max_dim = len(data['u_vals_true'])

    u_vals = data['u_vals'][:max_dim]
    v_vals = data['v_vals'][:max_dim]

    u_ent = max_dim * log(2*torch.pi*torch.e) / 2 + torch.log(u_vals).sum() / 2
    v_ent = max_dim * log(2*torch.pi*torch.e) / 2 + torch.log(v_vals).sum() / 2

    print(f'Input (u) estimated total variance: {float(u_vals.sum()):.05g}')
    print(f'Output (v) estimated total variance: {float(v_vals.sum()):.05g}')
    print(f'Input (u) estimated total entropy: {float(u_ent):.05g}')
    print(f'Output (v) estimated total entropy: {float(v_ent):.05g}')

    axes.plot(torch.sort(u_vals, descending=True).values, label='$u$ est.')
    if 'u_vals_true' in data:
        u_vals_true = data['u_vals_true']
        axes.plot(torch.sort(u_vals_true, descending=True).values, label='$u$ true')
        u_ent_true = max_dim * log(2*torch.pi*torch.e) / 2 + torch.log(u_vals_true).sum() / 2
        print(f'Input (u) computed total variance: {float(u_vals_true.sum()):.05g}')
        print(f'Input (u) computed total entropy: {float(u_ent_true):.05g}')

    axes.plot(torch.sort(data['v_vals'][:max_dim], descending=True).values, label='$v$ est.')
    if 'v_vals_true' in data:
        v_vals_true = data['v_vals_true']
        axes.plot(torch.sort(v_vals_true, descending=True).values, label='$v$ true')
        v_ent_true = max_dim * log(2*torch.pi*torch.e) / 2 + torch.log(v_vals_true).sum() / 2
        print(f'Output (v) computed total variance: {float(v_vals_true.sum()):.05g}')
        print(f'Output (v) computed total entropy: {float(v_ent_true):.05g}')

    axes.legend()

    mlx.show_and_save(fig, 'total_variance', config, name)
