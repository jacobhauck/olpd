import mlx
import torch.utils.data
import matplotlib.pyplot as plt
import os
import set_fonts

from modules.data import NormalizedOLDataset
from .pd2d import PD2DTrainer
from operatorlearning.modules import FunctionalL2Loss
from operatorlearning.data import OLDataset


@mlx.experiment
def plot_result(config, name, group=None):
    run = mlx.load_run(config['run_id'])
    run.config['device'] = config['device']
    trainer = PD2DTrainer(run.config, run)

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

    dataset = OLDataset(config['dataset'])
    if run.config['training'].get('normalize', False):
        NormalizedOLDataset(
            dataset,
            u_mean=trainer.datasets['train'].u_mean,
            u_std=trainer.datasets['train'].u_std,
            v_mean=trainer.datasets['train'].v_mean,
            v_std=trainer.datasets['train'].v_std
        )

    rel_l2 = FunctionalL2Loss(relative=True, squared=False)

    sub_path = os.path.relpath(config['dataset'], 'data')[:-len('.ol.h5')]
    trainer.model.train(False)
    for i in mlx.subset_indices(config, dataset):
        u, x, v, y = dataset[i]
        d = config['device']
        u, x, v, y = u.to(d)[None], x.to(d)[None], v.to(d)[None], y.to(d)[None]

        with torch.no_grad():
            v_pred = trainer.apply_model(u, x, y)

        if transform is not None:
            v, y = transform(v, y)

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

        fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(7, 5))
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
        axes[1][0].set_title('Predicted $x$ displacement')

        last = axes[1][1].imshow(v_pred[:, :, 1].T.cpu(), **im_kwargs)
        axes[1][1].set_title(f'Pred. $y$ disp. (m)')
        axes[1][1].set_xlabel('$x$')
        axes[1][1].set_aspect('equal')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
        fig.colorbar(last, cax=cbar_ax, label='Displacement (m)')

        mlx.show_and_save(fig, f'pred-{i}', config, name, sub_path)

        fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(5, 3))
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
        fig.colorbar(last, cax=cbar_ax, label=f'Error ($RL^2 = ${error:.2%})')

        mlx.show_and_save(fig, f'error-{i}', config, name, sub_path)
