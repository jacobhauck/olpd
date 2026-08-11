import mlx
import torch.utils.data
import matplotlib.pyplot as plt
import os
import set_fonts
from operatorlearning import OLDataset

from .wave1d import Wave1DTrainer


@mlx.experiment
def plot_results(config, name, group=None):
    run = mlx.load_run(config['run_id'])
    run.config['device'] = config['device']
    trainer = Wave1DTrainer(run.config, run)

    output_dir = os.path.join('results', name, run.name + '-' + run.id)
    os.makedirs(output_dir, exist_ok=True)
    rel_l2 = mlx.modules.RelativeL2Loss(squared=False)

    if config['dataset'] in trainer.datasets:
        dataset = trainer.datasets[config['dataset']]
    else:
        dataset = OLDataset(config['dataset'])

    trainer.model.train(False)
    d = config['device']
    for i in mlx.subset_indices(config, dataset):
        u, x, v, y = dataset[i]
        u, x, v, y = u.to(d), x.to(d), v.to(d), y.to(d)
        with torch.no_grad():
            v_pred = trainer.apply_model(u[None], x[None], y[None])[0]

        error = rel_l2(v[None], v_pred[None])

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].set_title('Final displacement')
        axes[0].set_xlabel('$x$')
        axes[0].set_ylabel('$v_1(x)$')
        axes[0].plot(y[:, 0].cpu(), v[:, 0].cpu(), label='True')
        axes[0].plot(y[:, 0].cpu(), v_pred[:, 0].cpu(), label=f'Pred (error = {100*error.item():.02f}%)')
        axes[0].legend()

        axes[1].set_title(f'Final velocity')
        axes[1].set_xlabel('$x$')
        axes[1].set_ylabel('$v_2(x)$')
        axes[1].plot(y[:, 0].cpu(), v[:, 1].cpu())
        axes[1].plot(y[:, 0].cpu(), v_pred[:, 1].cpu())

        plt.savefig(
            os.path.join(output_dir, f'{i}.{config.get("format", "png")}'),
            bbox_inches='tight'
        )

        if config['show']:
            plt.show()

        plt.close(fig)
