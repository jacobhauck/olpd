import os

import matplotlib.pyplot as plt
import mlx
import torch
from operatorlearning import GridFunction

from .crack1 import Crack1Trainer


@mlx.experiment
def run_experiment(config, name, group=None):
    run = mlx.load_run(config['run_id'])
    run.config['device'] = config['device']
    trainer = Crack1Trainer(run.config, run, no_data=True)

    output_dir = os.path.join('results', name, config['run_id'])
    os.makedirs(output_dir, exist_ok=True)

    y = GridFunction.uniform_x(
        torch.tensor(config['x_min']),
        torch.tensor(config['x_max']),
        torch.tensor(config['resolution'])
    )  # (*shape, 2)

    with torch.no_grad():
        v_basis = trainer.model.reconstructor_net(y[None])[0]  # (*shape, q, 1)

    q = v_basis.shape[-2]
    print(f'Model has {q} output basis functions')
    im_kwargs = {
        'vmin': -1.0,
        'vmax': 1.0,
        'cmap': 'seismic',
        'extent': [
            config['x_min'][0], config['x_max'][0],
            config['x_min'][1], config['x_max'][1]
        ],
        'origin': 'lower'
    }

    for j in range(q):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(v_basis[:, :, j, 0].cpu().T, **im_kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Output basis fn {j}')
        fig.colorbar(im)
        fig.savefig(os.path.join(output_dir, f'basis_{j}.png'), bbox_inches='tight')
        plt.close(fig)

    print(f'Wrote {q} images to {output_dir}')
