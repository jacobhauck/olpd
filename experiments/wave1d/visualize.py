import mlx
import mlx.utils
from operatorlearning.data import OLDataset, OLDatasetLibrary
import matplotlib.pyplot as plt
import os


@mlx.experiment
def visualize(config, name, group=None):
    data_lib = OLDatasetLibrary('wave1d')
    dataset = OLDataset(data_lib.dataset_path(config['split'], config['dataset_id']), stream_uv=False)

    output_dir = os.path.join('results', name, str(config['dataset_id']))
    os.makedirs(output_dir, exist_ok=True)

    for i in mlx.utils.subset_indices(config, dataset):
        u, x, v, y = dataset[i]
        fig, axes = plt.subplots(2, 2, figsize=(16, 6))
        axes[0][0].plot(x, u[:, 0], label='$u_1$')
        axes[0][0].set_title(f'Initial displacement ({i})')
        axes[0][0].set_xlabel('$x$')
        axes[0][0].set_ylabel('$u(x, 0)$')
        axes[0][0].legend()

        axes[1][0].plot(x, u[:, 1], label='$u_2$')
        axes[1][0].set_title(f'Initial velocity ({i})')
        axes[1][0].set_xlabel('$x$')
        axes[1][0].set_ylabel('$u_t(x, 0)$')
        axes[1][0].legend()

        axes[0][1].plot(y, v[:, 0], label='$v_1$')
        axes[0][1].set_title(f'Final displacement ({i})')
        axes[0][1].set_xlabel('$x$')
        axes[0][1].set_ylabel('$v(x, 0)$')
        axes[0][1].legend()

        axes[1][1].plot(y, v[:, 1], label='$v_2$')
        axes[1][1].set_title(f'Final velocity ({i})')
        axes[1][1].set_xlabel('$x$')
        axes[1][1].set_ylabel('$v_t(x, 0)$')
        axes[1][1].legend()

        plt.savefig(
            os.path.join(output_dir, config['split'] + '-' + str(i) + '.png'),
            bbox_inches='tight'
        )

        if config['show']:
            plt.show()

        plt.close(fig)

    print(f'Generated {len(mlx.utils.subset_indices(config, dataset))} visualizations in {output_dir}')
