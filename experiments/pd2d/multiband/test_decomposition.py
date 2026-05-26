import os

import matplotlib.pyplot as plt
import mlx
import torch
from operatorlearning.data import OLDataset


class TestMultibandDecomposition(mlx.Experiment):
    def run(self, config, name, group=None):
        dataset = OLDataset(config['dataset'], stream_uv=True)
        decompose = mlx.create_module(config['decomposition'])

        output_dir = os.path.join('results', name)
        os.makedirs(output_dir, exist_ok=True)

        errors = [[] for _ in range(decompose.num_steps)]
        decompose.to(config['device'])
        for u, x, v, y in dataset:
            d = config['device']
            v, y = v.to(d), y.to(d)
            v_d, _ = decompose(v[None], y[None])
            for b in range(decompose.num_steps):
                v_d_b = v_d.clone()
                v_d_b[:, b + 1:] = 0.0  # Zero bands above b
                v_recon = decompose.recompose(v_d_b)[0]
                errors[b].append((v_recon - v).abs().mean().item() / v.abs().mean().item())
        errors = torch.tensor(errors)

        for b in range(decompose.num_steps):
            print(f'Error for band {b}')
            print(f'Mean error: {float(errors[b].mean()):.04f}')
            print(f'Std error: {float(errors[b].std()):.04f}')
            print(f'Median error: {float(errors[b].median()):.04f}')
            print()

        decompose.to('cpu')
        for i, (u, x, v, y) in enumerate(dataset):
            if i >= config.get('max_plots', float('inf')):
                break

            v_d, y_d = decompose(v[None], y[None])
            v_d, y_d = v_d[0], y_d[0]
            v_rebuild = decompose.recompose(v_d[None])[0]

            v_min = float(v.min())
            v_max = float(v.max())
            im_kwargs = {
                'vmin': v_min,
                'vmax': v_max,
                'cmap': 'seismic',
                'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
                'origin': 'lower'
            }
            fig, axes = plt.subplots(figsize=(6, 6))
            last = axes.imshow(v[:, :, 0].T, **im_kwargs)
            axes.set_title(f'Final $x$ displacement ({i}) original')
            axes.set_xlabel('$x$')
            axes.set_ylabel('$y$')
            axes.set_aspect('equal')

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

            fig, axes = plt.subplots(figsize=(6, 6))
            last = axes.imshow(v_rebuild[:, :, 0].T, **im_kwargs)
            axes.set_title(f'Final $x$ displacement ({i}) rebuilt')
            axes.set_xlabel('$x$')
            axes.set_ylabel('$y$')
            axes.set_aspect('equal')

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
            fig.colorbar(last, cax=cbar_ax, label='Displacement')

            plt.savefig(
                os.path.join(output_dir, str(i) + '_recon.png'),
                bbox_inches='tight'
            )

            if config['show']:
                plt.show()

            plt.close(fig)

            for step in range(v_d.shape[0]):
                fig, axes = plt.subplots(figsize=(6, 6))
                im_kwargs['extent'] = (
                    config['xo'],
                    decompose.dilation ** step * config['xn'],
                    config['yo'],
                    decompose.dilation ** step * config['yn']
                )
                last = axes.imshow(v_d[step, :, :, 0].T, **im_kwargs)
                axes.set_title(f'Final $x$ displacement ({i}) step {step}')
                axes.set_xlabel('$x$')
                axes.set_ylabel('$y$')
                axes.set_aspect('equal')

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
                fig.colorbar(last, cax=cbar_ax, label='Displacement')

                plt.savefig(
                    os.path.join(output_dir, str(i) + '_' + str(step) + '.png'),
                    bbox_inches='tight'
                )

                if config['show']:
                    plt.show()

                plt.close(fig)

        print(f'Generated {config.get("max_plots", len(dataset))} visualizations in {output_dir}')
