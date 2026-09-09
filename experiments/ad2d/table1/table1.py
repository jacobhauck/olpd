import mlx
import os
import csv
import tqdm
from ..ad2d import AD2DTrainer
from operatorlearning.data import OLDatasetLibrary


@mlx.experiment
def make_table(config, name, group=None):
    lib = OLDatasetLibrary('ad2d')
    k_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    rows = [['Model', 1, 5, 25, 125, 625]]
    for model, ids in config['runs_by_model'].items():
        row = [model, 0, 0, 0, 0, 0]
        print(f'Evaluating model: {model}')
        for run_id in tqdm.tqdm(ids):
            run = mlx.load_run(run_id)
            _, dataset_id, _ = lib.parse_path(run.config['data']['train']['file_name'])
            trainer = AD2DTrainer(run.config, run)
            losses, _ = trainer.evaluate([config['split']])
            loss = losses[config['split']][config['metric']]
            row[k_index[dataset_id] + 1] = float(loss.mean())
        rows.append(row)

    output_file = os.path.join(mlx.results_dir(name), 'table1.csv')
    with open(output_file, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
