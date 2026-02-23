import mlx
import torch.utils.data
import wandb
import matplotlib.pyplot as plt
import os
from .pd2d import PD2DTrainer


class TestErrorExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        api = wandb.Api()
        prefix = mlx.wandb_config['entity'] + '/' + mlx.wandb_config['project']
        run = api.run(prefix + '/' + config['run_id'])
        run.step = run.lastHistoryStep
        run.config['device'] = 'cpu'
        trainer = PD2DTrainer(run.config, run)

        for metric, definition in config.get('additional_metrics', {}).items():
            trainer.metrics_fns[metric] = mlx.create_module(definition)

        # Handle model interface compatibility
        if 'fno' in run.config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model(u)
        elif 'gnot' in run.config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model([(u, x)], y)
        elif 'pcanet' in run.config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model(u)

        losses, metrics = trainer.evaluate(('train', 'test'))

        for dataset, dataset_losses in losses.items():
            print(f'===== Loss for dataset: "{dataset}" =====')
            for loss_name, loss in dataset_losses.items():
                print(f'    === Loss: {loss_name} ===')
                print(f'    Mean: {loss.mean().item():.05f}')
                print(f'    Median: {loss.median().item():.05f}')
                print(f'    Std.: {loss.std().item():.05f}')

        for dataset, dataset_metrics in metrics.items():
            print(f'===== Metric for dataset: "{dataset}" =====')
            for metric_name, metric in dataset_metrics.items():
                print(f'    === Loss: {metric_name} ===')
                print(f'    Mean: {metric.mean().item():.05f}')
                print(f'    Median: {metric.median().item():.05f}')
                print(f'    Std.: {metric.std().item():.05f}')
