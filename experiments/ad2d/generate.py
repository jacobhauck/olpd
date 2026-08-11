import mlx
import torch
import tqdm
from operatorlearning.data import OLDataset, OLDatasetLibrary
from operatorlearning.modules.basis import FullFourierBasis2d
from math import ceil
from scipy.sparse import coo_array
from scipy.sparse.linalg import splu
import numpy as np


class COOBuilder:
    def __init__(self, shape):
        self.shape = shape
        self.coords = []
        for _ in range(len(shape)):
            self.coords.append([])
        self.data = []

    def __setitem__(self, coords, data):
        data = np.broadcast_to(data, coords[0].shape)
        assert len(coords) == len(self.coords)

        for i in range(len(coords)):
            self.coords[i].append(coords[i])

        self.data.append(data)

    def get_coo(self):
        data = np.concatenate(self.data)
        coords = np.stack([np.concatenate(c) for c in self.coords])
        return coo_array((data, coords), self.shape)


def build_matrix(c, k, shape, dx, dy, dt):
    b = COOBuilder((*shape, *shape))  # (m, m, m, m)
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    x = x.flatten()
    y = y.flatten()

    x_p = (x + 1) % shape[0]
    x_m = (x - 1 + shape[0]) % shape[0]
    y_p = (y + 1) % shape[1]
    y_m = (y - 1 + shape[1]) % shape[1]

    # c[0] psi_x
    b[x, y, x_p, y] = -dt * (c[0] / (2*dx))
    b[x, y, x_m, y] = -dt * (-c[0] / (2*dx))

    # c[1] psi_y
    b[x, y, x, y_p] = -dt * (c[1] / (2*dy))
    b[x, y, x, y_m] = -dt * (-c[1] / (2*dy))

    # k psi_xx
    b[x, y, x_m, y] = -dt * (k / (dx**2))
    b[x, y, x, y] = -dt * (-2*k / (dx**2))
    b[x, y, x_p, y] = -dt * (k / (dx**2))

    # k psi_yy
    b[x, y, x, y_m] = -dt * (k / (dy**2))
    b[x, y, x, y] = -dt * (-2*k / (dy**2))
    b[x, y, x, y_p] = -dt * (k / (dy**2))

    # Main diagonal
    b[x, y, x, y] = 1.0

    # Get COO as matrix
    b_coo = b.get_coo()
    m_coo = b_coo.reshape((shape[0]*shape[1], shape[0]*shape[1]))

    # Convert to CSC for use in spsolve
    return m_coo.tocsc()


@mlx.experiment
def generate(config, name, group=None):
    torch.set_default_dtype(torch.double)
    if 'seed' in config:
        torch.random.manual_seed(config['seed'])

    x = torch.linspace(config['x0'], config['x1'], config['mesh_size'])
    dx = (x[1] - x[0])
    x += dx / 2
    y = torch.linspace(config['y0'], config['y1'], config['mesh_size'])
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
    xy_shift = xy + torch.tensor([config['c'][0] * config['t_final'], config['c'][1] * config['t_final']])
    basis_val_shift = basis.eval_basis(xy_shift[None])[0, ..., 0]  # (d, m, m)

    gx = basis.kx() / (2 * torch.pi)  # (d)
    gy = basis.ky() / (2 * torch.pi)  # (d)
    sqrt_lam = config['alpha'] / (config['beta'] + gx**2 + gy**2) ** (config['gamma'] / 2)
    # (d)

    matrix1 = build_matrix(config['c'], config['k'], tuple(basis_val.shape[1:]), dx, dy, config['dt'])
    lu1 = splu(matrix1)
    matrix2 = build_matrix(config['c'], config['k'], tuple(basis_val.shape[1:]), dx, dy, 2/3*config['dt'])
    lu2 = splu(matrix2)
    time_steps = ceil(config['t_final'] / config['dt'])
    method = config.get('method', 'implicit2')

    data_lib = OLDatasetLibrary('ad2d')
    dataset_id = data_lib.create_dataset(
        cx=config['c'][0],
        cy=config['c'][1],
        k=config['k'],
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
            if config['k'] == 0:
                v =  (coef * basis_val_shift).sum(dim=0)  # (m, m)
            elif method == 'implicit2':
                v = solve_equation_implicit2(u, lu1, lu2, config['dt'], time_steps)
            elif method == 'implicit1':
                v = solve_equation_implicit1(u, lu1, config['dt'], time_steps)
            elif method == 'explicit':
                v = solve_equation(u, config['c'], config['k'], dx, dy, config['dt'], time_steps, config['device'])
            else:
                raise ValueError('Invalid method')
            all_v.append(v[..., None].to(torch.float32))

        output_file = data_lib.dataset_path(split_name, dataset_id)
        print(f'Saving dataset split at {output_file}')
        OLDataset.write(
            all_u, [x[:, None]], all_v, [x[:, None]],
            file_name=output_file,
            u_disc=torch.zeros(split_size, dtype=torch.long),
            v_disc=torch.zeros(split_size, dtype=torch.long)
        )


def solve_equation(u, c, k, dx, dy, dt, time_steps, device, f=None):
    psi = u.clone().to(device)
    psi_x = torch.empty_like(psi)
    psi_y = torch.empty_like(psi)
    psi_xx = torch.empty_like(psi)
    psi_yy = torch.empty_like(psi)
    t = 0

    for n in range(time_steps):
        psi_x[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dx)
        psi_x[0, :] = (psi[1, :] - psi[-1, :]) / (2 * dx)
        psi_x[-1, :] = (psi[0, :] - psi[-2, :]) / (2 * dx)

        psi_y[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dy)
        psi_y[:, 0] = (psi[:, 1] - psi[:, -1]) / (2 * dy)
        psi_y[:, -1] = (psi[:, 0] - psi[:, -2]) / (2 * dy)

        psi_xx[1:-1, :] = (psi[:-2, :] - 2*psi[1:-1, :] + psi[2:, :]) / (dx**2)
        psi_xx[0, :] = (psi[-1, :] - 2*psi[0, :] + psi[1, :]) / (dx**2)
        psi_xx[-1, :] = (psi[-2, :] - 2*psi[-1, :] + psi[0, :]) / (dx**2)

        psi_yy[:, 1:-1] = (psi[:, :-2] - 2*psi[:, 1:-1] + psi[:, 2:]) / (dy**2)
        psi_yy[:, 0] = (psi[:, -1] - 2*psi[:, 0] + psi[:, 1]) / (dy**2)
        psi_yy[:, -1] = (psi[:, -2] - 2*psi[:, -1] + psi[:, 0]) / (dy**2)

        if f is None:
            f_val = 0
        else:
            f_val = f(t)

        psi += dt * (c[0] * psi_x + c[1] * psi_y + k * (psi_xx + psi_yy) + f_val)
        t += dt

    return psi.cpu()


def solve_equation_implicit1(u, lu1, dt, time_steps, f=None):
    psi = u.numpy().reshape(-1)
    t = 0.0

    for n in range(time_steps):
        if f is None:
            psi = lu1.solve(psi)
        else:
            psi = lu1.solve(psi + dt * f(t + dt))

        t += dt

    return torch.from_numpy(psi).cpu().reshape(u.shape)


def solve_equation_implicit2(u, lu1, lu2, dt, time_steps, f=None):
    psi = u.numpy().reshape(-1)
    last_psi = psi.copy()
    if f is None:
        psi = lu1.solve(psi)  # Backward Euler on first step
    else:
        psi = lu1.solve(psi + dt * f(dt))

    t = dt

    for n in range(time_steps - 1):  # BDF2 on subsequent steps
        if f is None:
            next_psi = lu2.solve((4/3) * psi - last_psi/3)
        else:
            next_psi = lu2.solve((4/3) * psi - last_psi/3 + (2/3*dt) * f(t + dt))
        last_psi = psi
        psi = next_psi
        t += dt

    return torch.from_numpy(psi).cpu().reshape(u.shape)
