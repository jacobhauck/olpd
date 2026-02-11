import mlx
from operatorlearning.data import OLDataset
import matplotlib.pyplot as plt
import os


class VisualizePD1DDataset(mlx.Experiment):
    def run(self, config, name, group=None):
        dataset = OLDataset(config['dataset'], stream_uv=True)

        output_dir = os.path.join('results', name)
        os.makedirs(output_dir, exist_ok=True)

        for i, (u, x, v, y) in enumerate(dataset):
            if i >= config.get('max_plots', float('inf')):
                break

            v_min = min(float(u.min()), float(v.min()))
            v_max = max(float(u.max()), float(v.max()))
            im_kwargs = {
                'vmin': v_min,
                'vmax': v_max,
                'cmap': 'seismic'
            }

            fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 8))
            axes[0][0].imshow(x[:, :, 0], **im_kwargs)
            axes[0][0].set_title(f'Initial $x$ displacement ({i})')
            axes[0][0].set_xlabel('$x$')
            axes[0][0].set_ylabel('$y$')
            axes[0][0].set_aspect('equal')

            axes[0][1].imshow(u[:, :, 1], **im_kwargs)
            axes[0][1].set_title(f'Initial $y$ displacement ({i})')
            axes[0][1].set_xlabel('$x$')
            axes[0][1].set_ylabel('$y$')
            axes[0][1].set_aspect('equal')

            axes[1][0].imshow(v[:, :, 0], **im_kwargs)
            axes[1][0].set_title(f'Final $x$ displacement ({i})')
            axes[1][0].set_xlabel('$x$')
            axes[1][0].set_ylabel('$y$')
            axes[1][0].set_aspect('equal')

            last = axes[1][1].imshow(v[:, :, 1], **im_kwargs)
            axes[1][1].set_title(f'Final $y$ displacement ({i})')
            axes[1][1].set_xlabel('$x$')
            axes[1][1].set_ylabel('$y$')
            axes[1][1].set_aspect('equal')

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
            fig.colorbar(last, cax=cbar_ax, label='Displacement')

            plt.savefig(
                os.path.join(output_dir, str(i) + '.png'),
                bbox_inches='tight'
            )

            if config['show']:
                plt.show()

            plt.close(fig)

        print(f'Generated {config.get("max_plots", len(dataset))} visualizations in {output_dir}')
