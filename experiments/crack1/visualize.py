import mlx
from operatorlearning.data import OLDataset
import random
import matplotlib.pyplot as plt
import os


@mlx.experiment
def experiment(config, name, group=None):
    dataset = OLDataset(config['dataset'])

    num_samples = min(len(dataset), config['max_plots'])
    if 'samples' in config:
        samples = config['samples']
    elif config.get('random', False):
        samples = random.sample(range(len(dataset)), num_samples)
    else:
        samples = range(num_samples)

    for i in samples:
        u, x, v, y = dataset[i]

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(x[:, 0], u[:, 0], label='top traction')
        axes[0].plot(x[:, 0], u[:, 1], label='bot traction')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('rel. stress')
        axes[0].legend()
        axes[0].set_title(f'Boundary relative traction ({i})')

        v_min = min(float(u.min()), float(v.min()))
        v_max = max(float(u.max()), float(v.max()))
        im_kwargs = {
            'vmin': v_min,
            'vmax': v_max,
            'cmap': 'plasma',
            'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
            'origin': 'lower'
        }
        axes[1].imshow(v[:, :, 0].T, **im_kwargs)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title(f'Damage field ({i})')

        if config.get('show', False):
            plt.show()

        output_dir = os.path.join(f'results/{name}/{config["dataset"]}')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'{i}.png'))
