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

        fig, axes = plt.subplots(2, 1, figsize=(4, 4))
        axes[0].plot(x[:, 0], u[:, 0], label='Top traction')
        axes[0].plot(x[:, 0], u[:, 1], label='Bottom traction')
        if 'traction_range' in config:
            axes[0].set_ylim(config['traction_range'])
        axes[0].set_ylabel('rel. stress')
        axes[0].legend()
        axes[0].set_title(f'Boundary relative traction profile')

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
        axes[1].set_title(f'Damage field')

        if config.get('show', False):
            plt.show()

        output_dir = os.path.join(f'results/{name}/{config["dataset"]}')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'{i}.{file_format}'), bbox_inches='tight')

        plt.close(fig)
