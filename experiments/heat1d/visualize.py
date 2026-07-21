import mlx
import mlx.utils
from operatorlearning.data import OLDataset
import matplotlib.pyplot as plt
import os


@mlx.experiment
def visualize(config, name, group=None):
    dataset = OLDataset(config['dataset'], stream_uv=False)

    output_dir = os.path.join('results', name)
    os.makedirs(output_dir, exist_ok=True)

    for i in mlx.utils.subset_indices(config, dataset):
        u, x, v, y = dataset[i]
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(16, 6))
        axes[0].plot(x, u)
        axes[0].set_title(f'Initial displacement ({i})')
        axes[0].set_xlabel('$x$')
        axes[0].set_ylabel('$u(x, 0)$')
        axes[1].plot(y, v)
        axes[1].set_title(f'Final displacement ({i})')
        axes[1].set_xlabel('$x$')
        axes[1].set_ylabel('$u(x, T)$')

        plt.savefig(
            os.path.join(output_dir, str(i) + '.png'),
            bbox_inches='tight'
        )

        if config['show']:
            plt.show()

        plt.close(fig)

    print(f'Generated {len(mlx.utils.subset_indices(config, dataset))} visualizations in {output_dir}')
