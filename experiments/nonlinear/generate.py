import mlx
import torch
import tqdm
from math import ceil
from operatorlearning.modules.basis import FullFourierBasis2d
from operatorlearning.data import OLDatasetLibrary, OLDataset


def make_movie_callback(u):
    import matplotlib.pyplot as plt

    plt.ion()
    im = plt.imshow(
        u.T,
        cmap='seismic',
        vmin=-u.abs().max().item() * 1.2,
        vmax=u.abs().max().item() * 1.2
    )
    plt.colorbar(im)
    plt.pause(0.5)

    def callback(psi, n, t):
        im.set_data(psi.cpu().T)
        plt.title(f'n = {n}, t = {t:.02g}')
        plt.pause(0.01)

    return callback


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
        a=config['a'],
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
                cbf = None
                if config.get('make_movie', False):
                    cbf = make_movie_callback(u)
                v = solve_equation_explicit(u, lambda psi, *_: -config['a'] * psi**3, xy, dx, dy, config, callback=cbf)
                if config.get('make_movie', False):
                    import matplotlib.pyplot as plt
                    plt.close()
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


def solve_equation_explicit(u, f, xy, dx, dy, config, velocity=None, callback=None):
    """
    :param u: (size, size) initial condition
    :param f: forcing function mapping (psi, x, y) -> f(psi, x, y)
    :param xy: (size, size, 2) x and y coordinates
    :param dx: x step size
    :param dy: y step size
    :param config: run configuration
    :param velocity: Optional initial velocity. Zero is used if not provided
    :param callback: Callback function called on each step
    :return:
    """
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    u = u.to(config['device'])
    c2 = config['c'] ** 2
    dt = config['dt']
    num_steps = int(ceil(config['t_final'] / dt))

    psi = u
    psi_xx = u.clone()
    psi_yy = u.clone()

    # First step
    n = 0
    t = 0
    if callback is not None:
        callback(psi, n, t)
    laplace(psi, psi_xx, psi_yy, dx, dy)
    psi_last = psi.clone()
    f_val = f(psi, x, y, t)
    if velocity is None:
        velocity = torch.zeros_like(psi)
    else:
        velocity = velocity.to(psi.device)
    psi += velocity * dt + dt**2/2 * (c2 * (psi_xx + psi_yy) + f_val)
    n += 1
    t += dt
    if callback is not None:
        callback(psi, n, t)

    while n <= num_steps:
        temp = psi.clone()
        laplace(psi, psi_xx, psi_yy, dx, dy)
        f_val = f(psi, x, y, t)
        psi = 2*psi - psi_last + dt**2 * (c2 * (psi_xx + psi_yy) + f_val)
        psi_last = temp
        n += 1
        t += dt
        if callback is not None:
            callback(psi, n, t)

    return psi.cpu()
