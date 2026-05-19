import mlx
import torch
import os

import matplotlib.pyplot as plt
from .pd2d import PD2DTrainer
from operatorlearning.modules import FunctionalL2Loss


@mlx.experiment
def run_experiment(config, name, *_, **__):
    run = mlx.load_run(config['run_id'])
    run.config['device'] = config['device']

    trainer = PD2DTrainer(run.config, run)
    trainer.model.train(False)

    dataset = trainer.datasets['test']
    indices = mlx.subset_indices(config, dataset)

    rel_l2 = FunctionalL2Loss(relative=True, squared=False)

    output_dir = os.path.join('results', name, config['run_id'])
    os.makedirs(output_dir, exist_ok=True)

    for i in indices:
        u, x, v, y = dataset[i]
        d = config['device']
        u, x, v, y = u.to(d), x.to(d), v.to(d), y.to(d)

        errors = []
        for scale in config['scales']:
            with torch.no_grad():
                v_pred = trainer.model(u[None] * scale, x[None], y[None])[0]
            v_scale = v * scale
            errors.append(float(rel_l2(v_pred, v_scale)))

        print(f'Errors for sample {i}')
        print(errors)
        print()

        plt.plot(config['scales'], errors)
        plt.xlabel('Scale')
        plt.ylabel('Rel L^2 error')
        plt.title(f'Nonlinearity on sample {i}')
        if config.get('show', False):
            plt.show()
        plt.savefig(os.path.join(output_dir, f'error_{i}.png'), bbox_inches='tight')
        plt.close()
