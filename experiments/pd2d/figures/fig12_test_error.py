import mlx
import os
import json
from operatorlearning.data import OLDatasetLibrary
from ..pd2d import PD2DTrainer


@mlx.experiment
def calc_error(config, name, group=None):
    lib = OLDatasetLibrary('elastic2d')
    error = {}

    for run_id in config['runs']:
        run = mlx.load_run(run_id)
        p = run.config['model']['encoder_net']['p']
        _, dataset_id, _ = lib.parse_path(run.config['data']['train']['file_name'])
        gamma = lib[dataset_id]['gamma']

        if gamma not in error:
            error[gamma] = {'p': [], 'error': []}
        trainer = PD2DTrainer(run.config, run)
        losses, _ = trainer.evaluate(('test',))
        error[gamma]['error'].append(losses['test']['objective'].mean().item())
        error[gamma]['p'].append(p)

    with open(os.path.join(mlx.results_dir(name), 'error.json'), 'w') as f:
        json.dump(error, f)
