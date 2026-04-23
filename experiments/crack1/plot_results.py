import mlx
import torch.utils.data
import wandb
import matplotlib.pyplot as plt
import os

from modules.data import NormalizedOLDataset
from .crack1 import Crack1Trainer
from operatorlearning.data import OLDataset


class PlotResults(mlx.Experiment):
    def run(self, config, name, group=None):
        api = wandb.Api()
        prefix = mlx.wandb_config['entity'] + '/' + mlx.wandb_config['project']
        run = api.run(prefix + '/' + config['run_id'])
        run.step = run.lastHistoryStep
        run.config['device'] = config['device']
        trainer = Crack1Trainer(run.config, run)
        if 'transform' in config:
            transform = mlx.create_module(config['transform']).to(config['device'])
        else:
            transform = None

        # Handle model interface compatibility
        if 'fno' in run.config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model(u)
        elif 'gnot' in run.config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model([(u, x)], y)
        elif 'pcanet' in run.config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model(u)

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
            dataset_name = 'custom'

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=config.get('random', True)
        )

        output_dir = os.path.join('results', run.name + '-' + run.id)
        os.makedirs(output_dir, exist_ok=True)

        trainer.model.train(False)
        for i, (u, x, v, y) in enumerate(data_loader):
            if i >= config['max_plots']:
                break

            d = config['device']
            u, x, v, y = u.to(d), x.to(d), v.to(d), y.to(d)

            with torch.no_grad():
                v_pred = trainer.apply_model(u, x, y)

            if transform is not None:
                v, y = transform(v, y)

            u, x, v, y, v_pred = u[0], x[0], v[0], y[0], v_pred[0]

            v_min = min(float(u.min()), float(v.min()))
            v_max = max(float(u.max()), float(v.max()))
            im_kwargs = {
                'vmin': v_min,
                'vmax': v_max,
                'cmap': 'seismic',
                'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
                'origin': 'lower'
            }

            fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 8))
            axes[0].imshow(v[:, :, 0].T.cpu(), **im_kwargs)
            axes[0].set_title(f'Final damage field ({i})')
            axes[0].set_xlabel('$x$')
            axes[0].set_ylabel('$y$')
            axes[0].set_aspect('equal')

            last = axes[1].imshow(v_pred[:, :, 0].T.cpu(), **im_kwargs)
            axes[1].set_title(f'Pred damage field ({i})')
            axes[1].set_xlabel('$x$')
            axes[1].set_ylabel('$y$')
            axes[1].set_aspect('equal')

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
            fig.colorbar(last, cax=cbar_ax, label='Damage')

            plt.savefig(
                os.path.join(output_dir, dataset_name + '_pred_' + str(i) + '.png'),
                bbox_inches='tight'
            )

            if config['show']:
                plt.show()

            plt.close(fig)
