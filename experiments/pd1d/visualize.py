import mlx
from operatorlearning.data import OLDataset
import matplotlib.pyplot as plt
import os


class VisualizePD1DDataset(mlx.Experiment):
    def run(self, config, name, group=None):
        dataset = OLDataset(config['dataset'], stream_uv=False)

        output_dir = os.path.join('results', name)
        os.makedirs(output_dir, exist_ok=True)

        for i, (u, x, v, y) in enumerate(dataset):
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

        print(f'Generated {len(dataset)} visualizations in {output_dir}')
