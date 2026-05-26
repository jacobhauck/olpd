import mlx
from .pd1d import PD1DTrainer


@mlx.experiment
def run_test(config, *_, **__):
    run = mlx.load_run(config['run_id'])
    trainer = PD1DTrainer(run.config, run)

    if 'checkpoint' in config:
        trainer.load_checkpoint(config['checkpoint'])

    losses, metrics = trainer.evaluate(('train', 'test'))

    for dataset, dataset_losses in losses.items():
        print(f'===== Loss for dataset: "{dataset}" =====')
        for loss_name, loss in dataset_losses.items():
            print(f'    === Loss: {loss_name} ===')
            print(f'    Mean: {loss.mean().item():.05f}')
            print(f'    Median: {loss.median().item():.05f}')
            print(f'    Std.: {loss.std().item():.05f}')
            print()
        print()
    print()

    for dataset, dataset_metrics in metrics.items():
        print(f'===== Metric for dataset: "{dataset}" =====')
        for metric_name, metric in dataset_metrics.items():
            print(f'    === Loss: {metric_name} ===')
            print(f'    Mean: {metric.mean().item():.05f}')
            print(f'    Median: {metric.median().item():.05f}')
            print(f'    Std.: {metric.std().item():.05f}')
            print()
        print()
