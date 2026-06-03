import mlx
from .train_basis import BasisTrainer
from modules.reconstruction import ReconstructionLoss

rel_loss = ReconstructionLoss(relative=True, squared=False)

def metrics_u(prediction, data):
    basis_val = prediction  # (B, *shape, p, d_out)
    u, x, _, _ = data
    u, x = u.to(basis_val.device), x.to(basis_val.device)
    return {'rel_recon': rel_loss(basis_val, u, x)}


def metrics_v(prediction, data):
    basis_val = prediction  # (B, *shape, q, d_out)
    _, _, v, y = data
    v, y = v.to(basis_val.device), y.to(basis_val.device)
    return {'rel_recon': rel_loss(basis_val, v, y)}


@mlx.experiment
def eval_basis(config, name, group=None):
    run = mlx.load_run(config['run_id'])
    trainer = BasisTrainer(run.config, run)
    if run.config.get('target_dist', 'input') == 'input':
        trainer.metrics = metrics_u
    else:
        trainer.metrics = metrics_v

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
