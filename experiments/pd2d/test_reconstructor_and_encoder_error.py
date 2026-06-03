import mlx
import torch.utils.data
import os
from .pd2d import PD2DTrainer
from operatorlearning.data import OLDataset
from modules.reconstruction import ReconstructionLoss


rel_loss = ReconstructionLoss(relative=True, squared=False)

def make_metrics(model, device):
    def metrics(_, data):
        u, x, v, y = data
        u, x, v, y = u.to(device), x.to(device), v.to(device), y.to(device)
        encoder_basis = model.encoder_net(x)  # (B, *in_shape, p, u_d_out)
        recon_basis = model.reconstructor_net(y)  # (B, *out_shape, q, v_d_out)
        return {
            'rel_recon': rel_loss(encoder_basis, u, x),
            'rel_encode': rel_loss(recon_basis, v, y)
        }

    return metrics


@mlx.experiment
def run_test(config, name, group=None):
    run = mlx.load_run(config['run_id'])
    run.config['device'] = config['device']
    trainer = PD2DTrainer(run.config, run)
    if 'checkpoint' in config:
        trainer.load_checkpoint(config['checkpoint'])

    trainer.metrics = make_metrics(trainer.model, config['device'])

    for dataset_name in config.get('additional_datasets', ()):
        name = os.path.basename(dataset_name)
        trainer.datasets[name] = OLDataset(dataset_name)
        trainer.data_loaders[name] = torch.utils.data.DataLoader(
            trainer.datasets[name],
            batch_size=1,
            shuffle=False
        )

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
