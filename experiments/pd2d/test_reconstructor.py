import mlx
import torch.utils.data
import os
from .pd2d import PD2DTrainer
from operatorlearning.data import OLDataset
from modules.reconstruction import ReconstructionLoss


rel_loss = ReconstructionLoss(relative=True, squared=False)

def metrics(prediction, data):
    basis_val = prediction  # (B, *shape, p, d_out)
    _, _, v, y = data
    v, y = v.to(basis_val.device), y.to(basis_val.device)
    return {'rel_recon': rel_loss(basis_val, u, x)}


@mlx.experiment
def run_test(config, name, group=None):
    run = mlx.load_run(config['run_id'])
    trainer = PD2DTrainer(run.config, run)
    if 'checkpoint' in config:
        trainer.load_checkpoint(config['checkpoint'])

    for name, metric in config.get('additional_metrics', {}).items():
        trainer.metrics_fns[name] = mlx.create_module(metric).to(run.config['device'])

    for dataset_name in config.get('additional_datasets', ()):
        name = os.path.basename(dataset_name)
        trainer.datasets[name] = OLDataset(dataset_name)
        trainer.data_loaders[name] = torch.utils.data.DataLoader(
            trainer.datasets[name],
            batch_size=1,
            shuffle=False
        )

    # Handle model interface compatibility
    if 'fno' in run.config['model']['name'].lower():
        trainer.apply_model = lambda u, x, y: trainer.model(u)
    elif 'gnot' in run.config['model']['name'].lower():
        trainer.apply_model = lambda u, x, y: trainer.model([(u, x)], y)
    elif 'pcanet' in run.config['model']['name'].lower():
        trainer.apply_model = lambda u, x, y: trainer.model(u)

    losses, metrics = trainer.evaluate(tuple(trainer.datasets.keys()))

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
