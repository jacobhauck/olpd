import mlx
import torch
from operatorlearning import GridFunction
from operatorlearning.modules.basis import FullFourierBasis2d

import modules.reconstruction


@mlx.experiment
def test_reconstruction(config, name, group=None):
    no_int_loss_fn = modules.reconstruction.ReconstructionLoss()
    int_loss_fn = modules.reconstruction.ReconstructionLoss(config['integrator'])
    basis = FullFourierBasis2d(2, torch.zeros(2), torch.ones(2))
    x = GridFunction.uniform_x(basis.x_min, basis.x_max, 128)
    # (*shape, 2)
    basis_val = basis.eval_basis(x[None], torch.arange(basis.dimension))
    # (B, d, *shape, 1)
    basis_val = torch.transpose(basis_val, 1, -1)[:, 0]  # (B, *shape, d)
    basis_val = basis_val[..., None]  # (B, *shape, d, 1)

    u = 0.5 + x[..., 0:1]**(1/3) * (1 - x[..., 1:2])  # (*shape, 1)

    print(u.shape, x.shape, basis_val.shape)
    no_int_loss = no_int_loss_fn(basis_val, u[None], x[None])
    int_loss = int_loss_fn(basis_val, u[None], x[None])
    print(f'No integration loss: {no_int_loss.item()}, integration loss: {int_loss.item()}')
