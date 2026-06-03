import os

import matplotlib.pyplot as plt
import mlx
import torch.utils.data
import set_fonts

from .crack1 import Crack1Trainer


@mlx.experiment
def superimpose(config, name, group=None):
    run = mlx.load_run(config['run_id'])
    run.config['device'] = config['device']
    trainer = Crack1Trainer(run.config, run)

    dataset = trainer.datasets[config['from_dataset']]

    output_dir = os.path.join('results', name, run.name + '-' + run.id)
    os.makedirs(output_dir, exist_ok=True)
    file_format = config.get('format', 'png')

    trainer.model.train(False)
    im_kwargs = {
        'vmin': 0.0,
        'vmax': 0.7,
        'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
        'origin': 'lower'
    }
    indices = list(map(int, torch.randperm(len(dataset))[:config['max_plots']]))
    for i in indices:
        d = config['device']
        u, x, v, y = dataset[i]
        u, x, v, y = u.to(d), x.to(d), v.to(d), y.to(d)

        with torch.no_grad():
            v_pred = trainer.model(u[None], x[None], y[None])[0]

        fig, ax = plt.subplots(figsize=(8, 2.5))
        im = ax.imshow(v[:, :, 0].T.cpu(), cmap='binary', **im_kwargs)
        im2 = ax.imshow(v_pred[:, :, 0].T.cpu(), cmap='Reds', alpha=config['alpha'], **im_kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Predicted (red) crack')
        fig.colorbar(im, label='True damage')
        fig.colorbar(im2, label='Pred. damage')
        if config.get('show', False):
            plt.show()
        fig.savefig(os.path.join(output_dir, f'compare_{i}.{file_format}'), bbox_inches='tight')
        plt.close(fig)
