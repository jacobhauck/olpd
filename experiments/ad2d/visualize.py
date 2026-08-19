import mlx
import mlx.utils
from operatorlearning.data import OLDataset, OLDatasetLibrary
import matplotlib.pyplot as plt
import os


@mlx.experiment
def visualize(config, name, group=None):
    lib = OLDatasetLibrary('ad2d')
    path = lib.dataset_path(config['split'], config['dataset_id'], config.get('resolution'))
    dataset = OLDataset(path, stream_uv=False)

    output_dir = os.path.join('results', name, str(config['dataset_id']))
    os.makedirs(output_dir, exist_ok=True)

    suffix = f'.{config.get("format", "png")}'
    if config.get('resolution') is not None:
        suffix = f'@{config["resolution"]}{suffix}'

    im_kwargs = {
        'cmap': 'seismic',
        'origin': 'lower'
    }
    for i in mlx.utils.subset_indices(config, dataset):
        u, x, v, y = dataset[i]
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(16, 6))
        im_kwargs['vmin'] = -float(u.abs().max())
        im_kwargs['vmax'] = -im_kwargs['vmin']
        axes[0].imshow(u[:, :, 0].T, **im_kwargs)
        axes[0].set_title(f'Initial state ({i})')
        axes[0].set_xlabel('$x$')
        axes[0].set_ylabel('$y$')
        axes[1].imshow(v[:, :, 0].T, **im_kwargs)
        axes[1].set_title(f'Final state ({i})')
        axes[1].set_xlabel('$x$')
        axes[1].set_ylabel('$y$')

        plt.savefig(
            os.path.join(output_dir, f'{config["split"]}-{i}{suffix}'),
            bbox_inches='tight'
        )

        if config['show']:
            plt.show()

        plt.close(fig)

    print(f'Generated {len(mlx.utils.subset_indices(config, dataset))} visualizations in {output_dir}')
