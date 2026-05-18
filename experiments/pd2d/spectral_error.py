import mlx
import torch
import numpy as np
from operatorlearning.modules.basis import FullFourierBasis2d
from operatorlearning.data import OLDataset
from .pd2d import PD2DTrainer
import matplotlib.pyplot as plt


def var_curve(basis, config):
    kx = basis.kx()
    ky = basis.ky()
    gx = kx / (2 * torch.pi)
    gy = ky / (2 * torch.pi)
    var = config['alpha']**2 / (config['beta'] + gx**2 + gy**2) ** config['gamma']
    return 1 - np.cumsum(sorted((var / var.sum()).cpu().numpy(), reverse=True))


@mlx.experiment
def run_experiment(config, name, group=None):
    basis = FullFourierBasis2d(config['num_modes'], config['x_min'], config['x_max'])
    basis.to(config['device'])

    # Sort basis functions by variance in training data
    kx = basis.kx()
    ky = basis.ky()
    gx = kx / (2 * torch.pi)
    gy = ky / (2 * torch.pi)
    var = config['alpha']**2 / (config['beta'] + gx**2 + gy**2) ** config['gamma']
    var_order = torch.argsort(var, descending=True)

    run = mlx.load_run(config['run_id'])
    trainer = PD2DTrainer(run.config, run, no_data=True)
    trainer.model.to(config['device']).train(False)
    dataset = OLDataset(config['dataset'])

    sample_indices = mlx.subset_indices(config, dataset)
    metric = mlx.create_module(config['metric']).to(config['device'])

    for i in sample_indices:
        u, x, v, y = dataset[i]
        d = config['device']
        u, x, v, y = u.to(d), x.to(d), v.to(d), y.to(d)

        v_coef = basis.coefficients(v[None], y[None])[0]
        with torch.no_grad():
            v_pred = trainer.model(u[None], x[None], y[None])[0]
        v_pred_coef = basis.coefficients(v_pred[None], y[None])[0]
        error = (v_coef - v_pred_coef) ** 2

        plt.plot((error[var_order] / error.sum()).cpu(), color='red')
        plt.plot((var[var_order] / var.sum()).cpu(), color='blue')
        plt.title('Projected error and (input) variance')
        plt.xlabel('basis function')
        plt.ylabel('Relative error/variance')
        plt.show()

        plt.plot((error[var_order] * var.sum() / (var[var_order] * error.sum())).cpu(), color='green')
        plt.title('Projected error / variance')
        plt.xlabel('basis functions')
        plt.ylabel('Ratio of projected error/variance')
        plt.show()
