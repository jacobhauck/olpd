import mlx
import torch
import tqdm
from math import ceil
from operatorlearning.modules.basis import FullFourierBasis2d
from operatorlearning.data import OLDatasetLibrary, OLDataset


@mlx.experiment
def generate(config, name, group=None):
    torch.set_default_dtype(torch.double)
    if 'seed' in config:
        torch.random.manual_seed(config['seed'])

    x = torch.linspace(config['x0'], config['x1'], config['mesh_size'] + 1)[:-1]
    dx = (x[1] - x[0])
    x += dx / 2
    y = torch.linspace(config['y0'], config['y1'], config['mesh_size'] + 1)[:-1]
    dy = (y[1] - y[0])
    y += dy / 2

    basis = FullFourierBasis2d(
        num_modes=config['num_modes'],
        x_min=(config['x0'], config['y0']),
        x_max=(config['x1'], config['y1'])
    )

    xy = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
    # (m, m, 2)
    basis_val = basis.eval_basis(xy[None])[0, ..., 0]  # (d, m, m)

    gx = basis.kx() / (2 * torch.pi)  # (d)
    gy = basis.ky() / (2 * torch.pi)  # (d)
    sqrt_lam = config['alpha'] / (config['beta'] + gx**2 + gy**2) ** (config['gamma'] / 2)
    # (d)

    method = config.get('method', 'explicit')

    data_lib = OLDatasetLibrary('nonlinear')
    dataset_id = data_lib.create_dataset(
        c=config['c'],
        t_final=config['t_final'],
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        num_modes=config['num_modes'],
        x0=config['x0'],
        x1=config['x1'],
        y0=config['y0'],
        y1=config['y1'],
        mesh_size=config['mesh_size']
    )

    for split_name, split_size in config['splits'].items():
        print(f'Generating split {split_name}')
        all_u = []
        all_v = []

        for _ in tqdm.tqdm(range(split_size)):
            coef = (torch.randn(sqrt_lam.shape) * sqrt_lam)[:, None, None]  # (d, 1, 1)
            u = (coef * basis_val).sum(dim=0)  # (m, m)
            all_u.append(u[..., None].to(torch.float32))
            if method == 'explicit':
                v = solve_equation_explicit(u, dx, dy, config)
            else:
                raise ValueError('Invalid method')
            all_v.append(v[..., None].to(torch.float32))

        output_file = data_lib.dataset_path(split_name, dataset_id)
        print(f'Saving dataset split at {output_file}')
        OLDataset.write(
            all_u, [xy], all_v, [xy],
            file_name=output_file,
            u_disc=torch.zeros(split_size, dtype=torch.long),
            v_disc=torch.zeros(split_size, dtype=torch.long)
        )


def laplace(psi, psi_xx, psi_yy, dx, dy):
    psi_xx[1:-1, :] = (psi[:-2, :] - 2 * psi[1:-1, :] + psi[2:, :]) / (dx ** 2)
    psi_xx[0, :] = (psi[-1, :] - 2 * psi[0, :] + psi[1, :]) / (dx ** 2)
    psi_xx[-1, :] = (psi[-2, :] - 2 * psi[-1, :] + psi[0, :]) / (dx ** 2)

    psi_yy[:, 1:-1] = (psi[:, :-2] - 2 * psi[:, 1:-1] + psi[:, 2:]) / (dy ** 2)
    psi_yy[:, 0] = (psi[:, -1] - 2 * psi[:, 0] + psi[:, 1]) / (dy ** 2)
    psi_yy[:, -1] = (psi[:, -2] - 2 * psi[:, -1] + psi[:, 0]) / (dy ** 2)


def solve_equation_explicit(u, dx, dy, config):
    """
    :param u: (size, size) initial condition
    :param dx: x step size
    :param dy: y step size
    :param config: run configuration
    :return:
    """
    u = u.to(config['device'])
    c2 = config['c'] ** 2
    dt = config['dt']
    num_steps = int(ceil(config['t_final'] / dt))

    psi = u
    psi_xx = u.clone()
    psi_yy = u.clone()

    # First step
    n = 0
    laplace(psi, psi_xx, psi_yy, dx, dy)
    psi_last = psi.clone()
    psi += dt**2/2 * (c2 * (psi_xx + psi_yy) - psi**3)
    n += 1

    while n <= num_steps:
        temp = psi.clone()
        laplace(psi, psi_xx, psi_yy, dx, dy)
        psi = 2*psi - psi_last + dt**2 * (c2 * (psi_xx + psi_yy) - psi**3)
        psi_last = temp
        n += 1

    return psi.cpu()
