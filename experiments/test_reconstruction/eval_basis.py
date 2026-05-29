import mlx
from .train_basis import BasisTrainer
from modules.reconstruction import ReconstructionLoss, coefficients

rel_loss = ReconstructionLoss(relative=True, squared=False)

def metrics(prediction, data):
    basis_val = prediction  # (B, *shape, p, d_out)
    u, x, _, _ = data
    u, x = u.to(basis_val.device), x.to(basis_val.device)
    return {'rel_recon': rel_loss(basis_val, u, x)}


@mlx.experiment
def eval_basis(config, name, group=None):
    run = mlx.load_run(config['run_id'])
    trainer = BasisTrainer(run.config, run)
    trainer.metrics = metrics

    losses, metrics_ = trainer.evaluate(datasets=('train', 'test',))

    for dataset, dataset_losses in losses.items():
        print(f'===== Loss for dataset: "{dataset}" =====')
        for loss_name, loss in dataset_losses.items():
            print(f'    === Loss: {loss_name} ===')
            print(f'    Mean: {loss.mean().item():.05f}')
            print(f'    Median: {loss.median().item():.05f}')
            print(f'    Std.: {loss.std().item():.05f}')

    for dataset, dataset_metrics in metrics_.items():
        print(f'===== Metric for dataset: "{dataset}" =====')
        for metric_name, metric in dataset_metrics.items():
            print(f'    === Loss: {metric_name} ===')
            print(f'    Mean: {metric.mean().item():.05f}')
            print(f'    Median: {metric.median().item():.05f}')
            print(f'    Std.: {metric.std().item():.05f}')
