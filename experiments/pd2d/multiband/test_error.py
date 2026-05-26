import mlx
import wandb
from .multiband import Multiband2dTrainer


class TestErrorExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        api = wandb.Api()
        prefix = mlx.wandb_path()
        run = api.run(prefix + '/' + config['run_id'])
        run.step = run.lastHistoryStep
        trainer = Multiband2dTrainer(run.config, run)

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