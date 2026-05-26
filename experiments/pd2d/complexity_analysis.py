import os

import mlx
import torch
import matplotlib.pyplot as plt
from operatorlearning.data import OLDataset
from operatorlearning.modules.basis import FullFourierBasis2d


@mlx.experiment
def run_experiment(config, name, group=None):
    dataset = OLDataset(config['dataset'])
    basis = FullFourierBasis2d(**config['basis']).to('cuda')

    if not os.path.exists('u_matrix.pt'):
        u_matrix = torch.empty((2 * basis.dimension, len(dataset)))
        v_matrix = torch.empty((2 * basis.dimension, len(dataset)))
        for i, (u, x, v, y) in enumerate(dataset):
            print('Computing coefficients for sample', i)
            u_matrix[:basis.dimension, i] = basis.coefficients(u[None, ..., 0:1].to('cuda'), x[None].to('cuda'))[0].cpu()
            u_matrix[basis.dimension:, i] = basis.coefficients(u[None, ..., 1:2].to('cuda'), x[None].to('cuda'))[0].cpu()
            v_matrix[:basis.dimension, i] = basis.coefficients(v[None, ..., 0:1].to('cuda'), x[None].to('cuda'))[0].cpu()
            v_matrix[basis.dimension:, i] = basis.coefficients(v[None, ..., 1:2].to('cuda'), x[None].to('cuda'))[0].cpu()
        torch.save(u_matrix, 'u_matrix.pt')
        torch.save(v_matrix, 'v_matrix.pt')
    else:
        u_matrix = torch.load('u_matrix.pt')
        v_matrix = torch.load('v_matrix.pt')

    u_mean = torch.mean(u_matrix, dim=1, keepdim=True)
    v_mean = torch.mean(v_matrix, dim=1, keepdim=True)
    p_u, d_u, q_u = torch.linalg.svd(u_matrix - u_mean)
    p_v, d_v, q_v = torch.linalg.svd(v_matrix - v_mean)
    p, d, q = torch.linalg.svd(p_u[:, :150].T @ p_v[:, :150])
    plt.plot(d**2, label='cor')
    plt.plot(d_u[:150]**2 / (d_u**2).sum(), label='var_u')
    plt.plot(d_v[:150]**2 / (d_v**2).sum(), label='var_v')
    plt.ylim(0, 1.5)
    plt.legend()
    plt.show()
