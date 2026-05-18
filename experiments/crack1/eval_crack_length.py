import os
import random

import matplotlib.pyplot as plt
import mlx
import torch
from mlx.experiments import load_run
from operatorlearning.data import OLDataset

from modules.data import NormalizedOLDataset
from .crack1 import Crack1Trainer


def crack_length(v, y, damage_thresh=0.3):
    """
    Calculates the length of a crack
    :param v: (nx, ny, 1) damage field
    :param y: (nx, ny, 2) coordinates
    :param damage_thresh: Threshold for considering damage
        to be part of the crack
    :return: length of the crack
    """
    is_cracked = (v[:, :, 0] > damage_thresh)
    cracked_x = y[..., 0][is_cracked]
    return float(cracked_x.max() - cracked_x.min())


@mlx.experiment
def run(config, name, group=None):
    run = load_run(config['run_id'])
    run.config['device'] = config['device']
    trainer = Crack1Trainer(run.config, run)
    trainer.model.train(False)

    # Handle model interface compatibility
    if 'fno' in run.config['model']['name'].lower():
        trainer.apply_model = lambda u, x, y: trainer.model(u)
    elif 'gnot' in run.config['model']['name'].lower():
        trainer.apply_model = lambda u, x, y: trainer.model([(u, x)], y)
    elif 'pcanet' in run.config['model']['name'].lower():
        trainer.apply_model = lambda u, x, y: trainer.model(u)

    dataset_name = config.get('from_dataset', 'test')
    if dataset_name in trainer.datasets:
        dataset = trainer.datasets[dataset_name]
    else:
        dataset = OLDataset(dataset_name)
        if run.config['training'].get('normalize', False):
            NormalizedOLDataset(
                dataset,
                u_mean=trainer.datasets['train'].u_mean,
                u_std=trainer.datasets['train'].u_std,
                v_mean=trainer.datasets['train'].v_mean,
                v_std=trainer.datasets['train'].v_std
            )
        dataset_name = 'custom'

    output_dir = os.path.join('results', name, run.name + '-' + run.id)
    os.makedirs(output_dir, exist_ok=True)
    plot_indices = frozenset(random.choices(range(len(dataset)), k=config['max_plots']))

    average_forces = []
    true_lengths = []
    pred_lengths = []

    for i, (u, x, v, y) in enumerate(dataset):
        v_pred = trainer.model(u[None], x[None], y[None])[0]
        true_length = crack_length(v, y, damage_thresh=config['damage_thresh'])
        pred_length = crack_length(v_pred, y, damage_thresh=config['damage_thresh'])

        true_lengths.append(true_length)
        pred_lengths.append(pred_length)
        average_forces.append(float(u.mean()))

        if i not in plot_indices:
            continue

        v_min = min(float(u.min()), float(v.min()))
        v_max = max(float(u.max()), float(v.max()))
        im_kwargs = {
            'vmin': v_min,
            'vmax': v_max,
            'cmap': 'plasma',
            'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
            'origin': 'lower'
        }

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(v[:, :, 0].T.cpu(), **im_kwargs)
        ax.set_title(f'Final damage field ({i}')
        ax.vlines([config['xo'], config['xo'] + true_length], config['yo'], config['yn'], colors=['green', 'green'])
        ax.vlines([config['xo'] + pred_length], config['yo'], config['yn'], colors=['blue'])

        if config.get('show', False):
            plt.show()

        fig.savefig(os.path.join(output_dir, dataset_name + '_' + str(i) + '.png'), bbox_inches='tight')
        plt.close(fig)

    true_lengths = torch.tensor(true_lengths)
    pred_lengths = torch.tensor(pred_lengths)
    plt.plot([min(true_lengths), max(true_lengths)], [min(true_lengths), max(true_lengths)], label='True')
    dist = (true_lengths - pred_lengths).abs() / 2**.5
    color = ['red' if d > config['good_cutoff'] else 'green' for d in dist]
    print('Bad indices')
    bad_indices = [i for i in range(len(color)) if color[i] == 'red']
    print(bad_indices)
    print('Traction range: ', min(average_forces[i] for i in bad_indices), max(average_forces[i] for i in bad_indices))
    plt.scatter(true_lengths, pred_lengths, color=color, label='Predicted')
    plt.xlabel('true length (m)')
    plt.ylabel('pred length (m)')
    plt.title('True and predicted crack lengths')
    plt.legend()
    plt.savefig(os.path.join(output_dir, dataset_name + '_line.png'), bbox_inches='tight')
    plt.show()

    plt.scatter(average_forces, true_lengths, color='green', label='True lengths')
    plt.scatter(average_forces, pred_lengths, color='orange', label='Pred lengths')
    plt.title('Crack length and average force')
    plt.xlabel('Average relative traction')
    plt.ylabel('Crack length (m)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, dataset_name + '_compare.png'), bbox_inches='tight')
    plt.show()
