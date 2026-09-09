import mlx
from operatorlearning.data import OLDataset
import matplotlib.pyplot as plt
import os
import set_fonts


@mlx.experiment
def experiment(config, name, group=None):
    dataset = OLDataset(config['dataset'])
    file_format = config.get('format', 'png')

    for i in mlx.subset_indices(config, dataset):
        u, x, v, y = dataset[i]

        im_kwargs = {
            'vmin': 0,
            'vmax': 1,
            'cmap': 'plasma',
            'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
            'origin': 'lower'
        }

        fig, ax = plt.subplots(figsize=(4, 6))

        ax.imshow(v[:, :, 0].T, **im_kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Final Damage field, v = {float(u):.03f}')

        if config.get('show', False):
            plt.show()

        output_dir = os.path.join(f'results/{name}/{config["dataset"]}')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'{i}.{file_format}'), bbox_inches='tight')

        plt.close(fig)
