import mlx
import torch.utils.data
import wandb
import matplotlib.pyplot as plt
import os
from .pd1d import PD1DTrainer


class PlotResults(mlx.Experiment):
    def run(self, config, name, group=None):
        api = wandb.Api()
        prefix = mlx.wandb_config['entity'] + '/' + mlx.wandb_config['project']
        run = api.run(prefix + '/' + config['run_id'])
        run.step = run.lastHistoryStep - 1
        trainer = PD1DTrainer(run.config, run)

        data_loader = torch.utils.data.DataLoader(
            trainer.datasets['test'],
            batch_size=1,
            shuffle=True
        )

        output_dir = os.path.join('results', name)
        os.makedirs(output_dir, exist_ok=True)
        rel_l2 = mlx.modules.RelativeL2Loss()

        for i, (u, x, v, y) in enumerate(data_loader):
            if i >= config['max_plots']:
                break

            trainer.model.train(False)
            trainer.model.to('cpu')
            with torch.no_grad():
                v_pred = trainer.apply_model(u, x, y)
            u, x, v, y, v_pred = u[0], x[0], v[0], y[0], v_pred[0]
            error = rel_l2(v, v_pred)

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
