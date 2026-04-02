import mlx
import torch.utils.data
import wandb
import matplotlib.pyplot as plt
import os
from .multiband import Multiband2dTrainer
from operatorlearning.modules import FunctionalL2Loss


class PlotResults(mlx.Experiment):
    def run(self, config, name, group=None):
        api = wandb.Api()
        run = api.run(mlx.wandb_path() + '/' + config['run_id'])
        run.step = run.lastHistoryStep
        run.config['device'] = config['device']
        trainer = Multiband2dTrainer(run.config, run)

        data_loader = torch.utils.data.DataLoader(
            trainer.datasets[config.get('from_dataset', 'test')],
            batch_size=1,
            shuffle=config.get('random', True)
        )

        output_dir = os.path.join('results', run.name + '-' + run.id)
        os.makedirs(output_dir, exist_ok=True)
        rel_l2 = FunctionalL2Loss(relative=True, squared=False)

        trainer.model.train(False)
        for j, (u, x, v, y) in enumerate(data_loader):
            if j >= config['max_plots']:
                break

            d = config['device']
            u, x, v, y = u.to(d), x.to(d), v.to(d), y.to(d)
            u, x = u[0], x[0],

            v_dec, y_dec = trainer.model.decomposition(v, y)

            for i in run.config['training']['bands']:
                with torch.no_grad():
                    v_pred = trainer.model.predict_band(u, x, y_dec[:, i], band=i)

                error = float(rel_l2(v_dec[:, i], v_pred))
                v, y, v_pred = v_dec[0, i], y_dec[0, i], v_pred[0]

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
                axes[0][0].imshow(v[:, :, 0].T.cpu(), **im_kwargs)
                axes[0][0].set_title(f'Final $x$ displacement ({i})')
                axes[0][0].set_xlabel('$x$')
                axes[0][0].set_ylabel('$y$')
                axes[0][0].set_aspect('equal')

                axes[0][1].imshow(v[:, :, 1].T.cpu(), **im_kwargs)
                axes[0][1].set_title(f'Final $y$ displacement ({i})')
                axes[0][1].set_xlabel('$x$')
                axes[0][1].set_ylabel('$y$')
                axes[0][1].set_aspect('equal')

                axes[1][0].imshow(v_pred[:, :, 0].T.cpu(), **im_kwargs)
                axes[1][0].set_title(f'Pred $x$ displacement ({i})')
                axes[1][0].set_xlabel('$x$')
                axes[1][0].set_ylabel('$y$')
                axes[1][0].set_aspect('equal')

                last = axes[1][1].imshow(v_pred[:, :, 1].T.cpu(), **im_kwargs)
                axes[1][1].set_title(f'Pred $y$ displacement ({i})')
                axes[1][1].set_xlabel('$x$')
                axes[1][1].set_ylabel('$y$')
                axes[1][1].set_aspect('equal')

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
                fig.colorbar(last, cax=cbar_ax, label='Displacement')

                plt.savefig(
                    os.path.join(output_dir, config.get('from_dataset', 'test') + f'_pred_{j}_band_{i}.png'),
                    bbox_inches='tight'
                )

                if config['show']:
                    plt.show()

                plt.close(fig)

                fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 8))
                axes[0].imshow((v - v_pred).abs()[:, :, 0].T.cpu(), **err_kwargs)
                axes[0].set_title(f'Error $x$ displacement ({i})')
                axes[0].set_xlabel('$x$')
                axes[0].set_ylabel('$y$')
                axes[0].set_aspect('equal')

                last = axes[1].imshow((v - v_pred).abs()[:, :, 1].T.cpu(), **err_kwargs)
                axes[1].set_title(f'Error $y$ displacement ({i})')
                axes[1].set_xlabel('$x$')
                axes[1].set_ylabel('$y$')
                axes[1].set_aspect('equal')
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
                fig.colorbar(last, cax=cbar_ax, label=f'Displacement error ($RL^2 = $ {100*error:.02f}%)')

                plt.savefig(
                    os.path.join(output_dir, config.get('from_dataset', 'test') + f'_error_{j}_band_{i}.png'),
                    bbox_inches='tight'
                )

                if config['show']:
                    plt.show()

                plt.close(fig)
