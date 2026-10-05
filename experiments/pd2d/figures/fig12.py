import mlx
import matplotlib.pyplot as plt
from operatorlearning.data import OLDataset, OLDatasetLibrary
from operatorlearning.modules import FunctionalL2Loss, SplineGridIntegrator

import set_fonts
import torch
from ..pd2d import PD2DTrainer


@mlx.experiment
def fig13(config, name, group=None):
    fig, axes = plt.subplots(1, 5, figsize=config['figure_size'])
    dataset = OLDataset(config['dataset'])
    lib = OLDatasetLibrary('elastic2d')
    u, x, v, y = dataset[config['sample']]

    im_kwargs = {
        'cmap': 'seismic',
        'origin': 'lower',
        'vmin': -v[..., 0].abs().max().item(),
        'vmax': v[..., 0].abs().max().item()
    }

    axes[0].imshow(v[:, :, 0].T, **im_kwargs)
    axes[0].set_title('Ground truth')
    axes[0].set_axis_off()

    l2_loss = FunctionalL2Loss(
        relative=True,
        squared=False,
        integrator={
            'name': 'SplineGridIntegrator',
            'n': 3,
            'x_min': [0.0, 0.0],
            'x_max': [5.0, 5.0]
        }
    )

    for i, run_id in enumerate(config['runs']):
        run = mlx.load_run(run_id)
        _, dataset_id, _ = lib.parse_path(run.config['data']['train']['file_name'])
        gamma = lib[dataset_id]['gamma']
        trainer = PD2DTrainer(run.config, run, no_data=True)
        trainer.model.train(False)
        d = run.config['device']
        with torch.no_grad():
            v_pred = trainer.apply_model(u[None].to(d), x[None].to(d), y[None].to(d))[0].cpu()

        error = l2_loss(v[None], v_pred[None])
        axes[i + 1].imshow(v_pred[:, :, 0].T, **im_kwargs)
        axes[i + 1].set_title(f'$\\gamma = {gamma}$ \n $\\textnormal{{err.}} = {error.item()*100:.01f}\\%$')
        axes[i + 1].set_axis_off()

    fig.tight_layout()

    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bbox0 = get_bbox(axes[0])
    bbox1 = get_bbox(axes[1])
    x_middle = (bbox0.x1 + bbox1.x0) / 2

    line = plt.Line2D(
        [x_middle, x_middle], [0.1, 0.9],
        transform=fig.transFigure, color='black', linestyle='--'
    )
    fig.add_artist(line)

    mlx.show_and_save(fig, 'generalization-sample', config, name)
