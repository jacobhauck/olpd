import os

import matplotlib.pyplot as plt
import mlx
import torch.utils.data

from .pd1d import PD1DTrainer


@mlx.experiment
def plot_results(config, name):
    mlx.configure_plotting(config)
    run = mlx.load_run(config['run_id'])
    run.config['device'] = 'cpu'
    trainer = PD1DTrainer(run.config, run)

    # Handle model interface compatibility
    if 'fno' in run.config['model']['name'].lower():
        trainer.apply_model = lambda u, x, y: trainer.model(u)
    elif 'gnot' in run.config['model']['name'].lower():
        trainer.apply_model = lambda u, x, y: trainer.model([(u, x)], y)

    data_loader = torch.utils.data.DataLoader(
        trainer.datasets['test'],
        batch_size=1,
        shuffle=True
    )

    output_dir = os.path.join('results', name)
    os.makedirs(output_dir, exist_ok=True)
    rel_l2 = mlx.modules.RelativeL2Loss()

    trainer.model.train(False)
    for i, (u, x, v, y) in enumerate(data_loader):
        if i >= config['max_plots']:
            break

        with torch.no_grad():
            v_pred = trainer.apply_model(u, x, y)
        error = rel_l2(v, v_pred)
        u, x, v, y, v_pred = u[0], x[0], v[0], y[0], v_pred[0]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].plot(x[:, 0], u[:, 0])
        axes[0].set_title(f'Initial displacement ({i})')
        axes[0].set_xlabel('$x$')
        axes[0].set_ylabel('$u(x, 0)$')
        axes[1].plot(y[:, 0], v[:, 0], label='True')
        axes[1].plot(y[:, 0], v_pred[:, 0], label='Pred')
        axes[1].set_title(f'Final displacement ({i}); error = {100*error.item():.02f}%')
        axes[1].set_xlabel('$x$')
        axes[1].set_ylabel('$u(x, T)$')
        axes[1].legend()

        plt.savefig(
            os.path.join(output_dir, str(i) + '.png'),
            bbox_inches='tight'
        )

        if config['show']:
            plt.show()

        plt.close(fig)
