import mlx
import matplotlib.pyplot as plt
from operatorlearning.data import OLDataset, OLDatasetLibrary
import set_fonts
import torch
from ..pd2d import PD2DTrainer


@mlx.experiment
def fig13(config, name, group=None):
    fig, axes = plt.subplots(1, 5, figsize=config['figure_size'])
    dataset = OLDataset(config['dataset'])
    lib = OLDatasetLibrary('elastic2d')
    u, x, v, y = dataset[config['sample']]

    axes[0].plot(y[:, config['y_index'], 0], v[:, config['y_index'], 0])
    axes[0].set_title('Ground truth')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$v(x, L/2)$')

    for i, run_id in enumerate(config['runs']):
        run = mlx.load_run(run_id)
        _, dataset_id, _ = lib.parse_path(run.config['data']['train']['file_name'])
        gamma = lib[dataset_id]['gamma']
        trainer = PD2DTrainer(run.config, run, no_data=True)
        trainer.model.train(False)
        d = run.config['device']
        with torch.no_grad():
            v_pred = trainer.apply_model(u[None].to(d), x[None].to(d), y[None].to(d))[0].cpu()

        axes[i + 1].plot(y[:, config['y_index'], 0], v_pred[:, config['y_index'], 0])
        axes[i + 1].set_xlabel('$x$')
        axes[i + 1].set_title(f'$\\gamma = {gamma}$')
        axes[i + 1].set_yticks([])

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

    mlx.show_and_save(fig, 'generalization-cross-section', config, name)
