import mlx
import torch.utils.data
import wandb
import matplotlib.pyplot as plt
import os
import set_fonts

from modules.data import NormalizedOLDataset
from modules.reconstruction import coefficients, ReconstructionLoss
from .pd2d import PD2DTrainer
from operatorlearning.modules import FunctionalL2Loss
from operatorlearning.data import OLDataset


class PlotResults(mlx.Experiment):
    def run(self, config, name, group=None):
        run = mlx.load_run(config['run_id'])
        run.config['device'] = config['device']
        trainer = PD2DTrainer(run.config, run)
        if 'transform' in config:
            transform = mlx.create_module(config['transform']).to(config['device'])
        else:
            transform = None

        dataset_name = config.get('from_dataset', 'test')
        if dataset_name in trainer.datasets:
            dataset = trainer.datasets[dataset_name]
        else:
            dataset = OLDataset(dataset_name)
            if run.config['training'].get('normalize', False):
                NormalizedOLDataset(
                    dataset,
                    u_mean=trainer.datasets['train'].u_mean,
                    u_std=trainer.datasets['train'].u_std,
                    v_mean=trainer.datasets['train'].v_mean,
                    v_std=trainer.datasets['train'].v_std
                )
            dataset_name = os.path.basename(dataset_name)

        output_dir = os.path.join('results', run.name + '-' + run.id)
        os.makedirs(output_dir, exist_ok=True)
        print(f'Using output directory {output_dir}')
        print(f'Dataset name {dataset_name}')
        rel_l2 = FunctionalL2Loss(relative=True, squared=False)
        rel_recon = ReconstructionLoss(relative=True, squared=False)
        file_format = '.' + config.get('format', 'png')

        trainer.model.train(False)
        for i in mlx.subset_indices(config, dataset):
            u, x, v, y = dataset[i]
            d = config['device']
            u, x, v, y = u.to(d)[None], x.to(d)[None], v.to(d)[None], y.to(d)[None]

            with torch.no_grad():
                v_pred = trainer.apply_model(u, x, y)
                v_basis = trainer.model.reconstructor_net(y)
                coef = coefficients(v_basis, v, y)
                v_best = torch.einsum('b...pd,bp->b...d', v_basis, coef)

            if transform is not None:
                v, y = transform(v, y)

            error = float(rel_l2(v, v_pred))
            error_best = float(rel_l2(v, v_best))
            try_error_best = float(rel_recon(v_basis, v, y))
            print('Best error (from FunctionalL2Loss)', error_best)
            print('Best error (from ReconstructionLoss)', try_error_best)
            u, x, v, y, v_pred, v_best = u[0], x[0], v[0], y[0], v_pred[0], v_best[0]

            v_min = min(float(u.min()), float(v.min()))
            v_max = max(float(u.max()), float(v.max()))
            im_kwargs = {
                'vmin': v_min,
                'vmax': v_max,
                'cmap': 'seismic',
                'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
                'origin': 'lower'
            }

            fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(4, 3.5))
            axes[0][0].imshow(v[:, :, 0].T.cpu(), **im_kwargs)
            axes[0][0].set_title(f'Final $x$ disp. (m)')
            axes[0][0].set_ylabel('$y$')
            axes[0][0].set_aspect('equal')

            axes[0][1].imshow(v[:, :, 1].T.cpu(), **im_kwargs)
            axes[0][1].set_title(f'Final $y$ disp. (m)')
            axes[0][1].set_aspect('equal')

            axes[1][0].imshow(v_pred[:, :, 0].T.cpu(), **im_kwargs)
            axes[1][0].set_title(f'Pred. $x$ disp. (m)')
            axes[1][0].set_xlabel('$x$')
            axes[1][0].set_ylabel('$y$')
            axes[1][0].set_aspect('equal')

            last = axes[1][1].imshow(v_pred[:, :, 1].T.cpu(), **im_kwargs)
            axes[1][1].set_title(f'Pred. $y$ disp. (m)')
            axes[1][1].set_xlabel('$x$')
            axes[1][1].set_aspect('equal')

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
            fig.colorbar(last, cax=cbar_ax, label=f'Displacement ($RL^2 = $ {100*error:.02f}%)')

            plt.savefig(
                os.path.join(output_dir, dataset_name + '_pred_' + str(i) + file_format),
                bbox_inches='tight'
            )

            if config['show']:
                plt.show()

            plt.close(fig)

            fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 8))
            axes[0].imshow(v_best[:, :, 0].T.cpu(), **im_kwargs)
            axes[0].set_title(f'Best $x$ displacement ({i})')
            axes[0].set_xlabel('$x$')
            axes[0].set_ylabel('$y$')
            axes[0].set_aspect('equal')

            last = axes[1].imshow(v_best[:, :, 1].T.cpu(), **im_kwargs)
            axes[1].set_title(f'Best $y$ displacement ({i})')
            axes[1].set_xlabel('$x$')
            axes[1].set_ylabel('$y$')
            axes[1].set_aspect('equal')
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
            fig.colorbar(last, cax=cbar_ax, label=f'Displacement (error = {100*error_best:.02f}%)')

            plt.savefig(
                os.path.join(output_dir, dataset_name + '_best_' + str(i) + file_format),
                bbox_inches='tight'
            )

            if config['show']:
                plt.show()

            plt.close(fig)
