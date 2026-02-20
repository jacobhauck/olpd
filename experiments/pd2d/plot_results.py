import mlx
import torch.utils.data
import wandb
import matplotlib.pyplot as plt
import os
from .pd2d import PD2DTrainer


class PlotResults(mlx.Experiment):
    def run(self, config, name, group=None):
        api = wandb.Api()
        prefix = mlx.wandb_config['entity'] + '/' + mlx.wandb_config['project']
        run = api.run(prefix + '/' + config['run_id'])
        run.step = run.lastHistoryStep - 1
        run.config['device'] = 'cpu'
        trainer = PD2DTrainer(run.config, run)

        # Handle model interface compatibility
        if 'fno' in run.config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model(u)
        elif 'gnot' in run.config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model([(u, x)], y)
        elif 'pcanet' in run.config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model(u)

        data_loader = torch.utils.data.DataLoader(
            trainer.datasets[config.get('from_dataset', 'test')],
            batch_size=1,
            shuffle=config.get('random', True)
        )

        output_dir = os.path.join('results', run.name)
        os.makedirs(output_dir, exist_ok=True)
        rel_l2 = mlx.modules.RelativeL2Loss()

        trainer.model.train(False)
        for i, (u, x, v, y) in enumerate(data_loader):
            if i >= config['max_plots']:
                break

            with torch.no_grad():
                v_pred = trainer.apply_model(u, x, y)

            error = float(rel_l2(v, v_pred))
            u, x, v, y, v_pred = u[0], x[0], v[0], y[0], v_pred[0]
            v_min = min(float(u.min()), float(v.min()))
            v_max = max(float(u.max()), float(v.max()))
            err_fn = (v - v_pred).abs()
            im_kwargs = {
                'vmin': v_min,
                'vmax': v_max,
                'cmap': 'seismic',
                'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
                'origin': 'lower'
            }
            err_kwargs = {
                'vmin': float(err_fn.min()),
                'vmax': float(err_fn.max()),
                'cmap': 'plasma',
                'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
                'origin': 'lower'
            }

            fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 8))
            axes[0][0].imshow(v[:, :, 0].T, **im_kwargs)
            axes[0][0].set_title(f'Final $x$ displacement ({i})')
            axes[0][0].set_xlabel('$x$')
            axes[0][0].set_ylabel('$y$')
            axes[0][0].set_aspect('equal')

            axes[0][1].imshow(v[:, :, 1].T, **im_kwargs)
            axes[0][1].set_title(f'Final $y$ displacement ({i})')
            axes[0][1].set_xlabel('$x$')
            axes[0][1].set_ylabel('$y$')
            axes[0][1].set_aspect('equal')

            axes[1][0].imshow(v_pred[:, :, 0].T, **im_kwargs)
            axes[1][0].set_title(f'Pred $x$ displacement ({i})')
            axes[1][0].set_xlabel('$x$')
            axes[1][0].set_ylabel('$y$')
            axes[1][0].set_aspect('equal')

            last = axes[1][1].imshow(v_pred[:, :, 1].T, **im_kwargs)
            axes[1][1].set_title(f'Pred $y$ displacement ({i})')
            axes[1][1].set_xlabel('$x$')
            axes[1][1].set_ylabel('$y$')
            axes[1][1].set_aspect('equal')

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
            fig.colorbar(last, cax=cbar_ax, label='Displacement')

            plt.savefig(
                os.path.join(output_dir, config.get('from_dataset', 'test') + '_pred_' + str(i) + '.png'),
                bbox_inches='tight'
            )

            if config['show']:
                plt.show()

            plt.close(fig)

            fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 8))
            axes[0].imshow((v - v_pred).abs()[:, :, 0].T, **err_kwargs)
            axes[0].set_title(f'Error $x$ displacement ({i})')
            axes[0].set_xlabel('$x$')
            axes[0].set_ylabel('$y$')
            axes[0].set_aspect('equal')

            last = axes[1].imshow((v - v_pred).abs()[:, :, 1].T, **err_kwargs)
            axes[1].set_title(f'Error $y$ displacement ({i})')
            axes[1].set_xlabel('$x$')
            axes[1].set_ylabel('$y$')
            axes[1].set_aspect('equal')
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
            fig.colorbar(last, cax=cbar_ax, label=f'Displacement error ($RL^2 = $ {100*error:.02f}%)')

            plt.savefig(
                os.path.join(output_dir, config.get('from_dataset', 'test') + '_error_' + str(i) + '.png'),
                bbox_inches='tight'
            )

            if config['show']:
                plt.show()

            plt.close(fig)
