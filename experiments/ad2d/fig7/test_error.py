import mlx
import json
import os
from operatorlearning.data import OLDatasetLibrary
from ..ad2d import AD2DTrainer


@mlx.experiment
def test_error(config, name, group=None):
    lib = OLDatasetLibrary('ad2d')
    ps = {1: [], 5: [], 25: [], 125: [], 625: []}
    errors = {1: [], 5: [], 25: [], 125: [], 625: []}

    for run_id in config['runs']:
        run = mlx.load_run(run_id)
        p = run.config['model']['encoder_net']['p']

        train_dataset = run.config['data']['train']['file_name']
        _, dataset_id, _ = lib.parse_path(train_dataset)
        k_factor = int(round(lib[dataset_id]['k'] / 1e-5))
        ps[k_factor].append(p)

        trainer = AD2DTrainer(run.config, run)
        losses, _ = trainer.evaluate([config['split']])
        loss = losses[config['split']]['objective']
        errors[k_factor].append(loss.mean().item())

    with open(os.path.join(mlx.results_dir(name), 'error.json'), 'w') as f:
        json.dump({'ps': ps, 'errors': errors}, f)
