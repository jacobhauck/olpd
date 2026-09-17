import mlx
import torch
import os
import tqdm
from operatorlearning.data import OLDatasetLibrary, OLDataset
from operatorlearning.modules.basis import FullFourierBasis2d


def compute_variance_heat1d(config):
    pi = torch.pi
    size = config['x1'] - config['x0']
    t = config['t_final']
    k = config['k']
    n = torch.arange(0, config['num_modes'] + 1).to(torch.float)  # (num_modes + 1,)
    sq_lam = config['alpha'] / (config['beta'] + n**2) ** (config['gamma'] / 2)

    u_coef_c_1 = sq_lam  # (num_modes + 1,)
    u_coef_c_2 = sq_lam  # (num_modes + 1,)
    u_coef_s_1 = sq_lam[1:]  # (num_modes,)
    u_coef_s_2 = sq_lam[1:]  # (num_modes,)

    u_var = torch.cat([u_coef_c_1, u_coef_c_2, u_coef_s_1, u_coef_s_2]) ** 2

    v_coef_c_1 = (sq_lam * torch.exp(-k * 4 * n ** 2 * pi ** 2 / size ** 2 * t))
    # (num_modes + 1,)
    v_coef_c_2 = (sq_lam * torch.exp(-k * 4 * n ** 2 * pi ** 2 / size ** 2 * t))
    # (num_modes + 1,)
    v_coef_s_1 = (sq_lam[1:] * torch.exp(-k * 4 * n[1:] ** 2 * pi ** 2 / size ** 2 * t))
    # (num_modes,)
    v_coef_s_2 = (sq_lam[1:] * torch.exp(-k * 4 * n[1:] ** 2 * pi ** 2 / size ** 2 * t))
    # (num_modes,)

    v_var = torch.cat([v_coef_c_1, v_coef_c_2, v_coef_s_1, v_coef_s_2]) ** 2

    return u_var, v_var


def compute_variance_wave1d(config):
    pi = torch.pi
    size = config['x1'] - config['x0']
    t = config['t_final']
    c = config['c']
    n = torch.arange(0, config['num_modes'] + 1).to(torch.float)  # (num_modes + 1,)
    sq_lam = config['alpha'] / (config['beta'] + n**2) ** (config['gamma'] / 2)

    u_coef_c_1 = sq_lam  # (num_modes + 1,)
    u_coef_c_2 = sq_lam  # (num_modes + 1,)
    u_coef_s_1 = sq_lam[1:]  # (num_modes,)
    u_coef_s_2 = sq_lam[1:]  # (num_modes,)

    u_var = torch.cat([u_coef_c_1, u_coef_c_2, u_coef_s_1, u_coef_s_2]) ** 2

    v_coef_c_1 = u_coef_c_1.clone()
    v_coef_c_1[1:] *= torch.cos(n[1:] * (2 * pi * c * t / size))
    v_coef_c_1[0] += t * u_coef_c_2[0]
    v_coef_c_1[1:] += (size / (n[1:] * 2 * pi * c)) * torch.sin(n[1:] * (2 * pi * c * t / size)) * u_coef_c_2[1:]

    v_coef_s_1 = u_coef_s_1.clone()
    v_coef_s_1 *= torch.cos(n[1:] * (2 * pi * c * t / size))
    v_coef_s_1 += (size / (n[1:] * 2 * pi * c)) * torch.sin(n[1:] * (2 * pi * c * t / size)) * u_coef_s_2

    v_coef_c_2 = u_coef_c_2.clone()
    v_coef_c_2[1:] *= torch.cos(n[1:] * (2 * pi * c * t / size))
    v_coef_c_2[1:] -= ((n[1:] * 2 * pi * c) / size) * torch.sin(n[1:] * (2 * pi * c * t / size)) * u_coef_c_1[1:]

    v_coef_s_2 = u_coef_s_2.clone()
    v_coef_s_2 *= torch.cos(n[1:] * (2 * pi * c * t / size))
    v_coef_s_2 -= ((n[1:] * 2 * pi * c) / size) * torch.sin(n[1:] * (2 * pi * c * t / size)) * u_coef_s_1

    v_var = torch.cat([v_coef_c_1, v_coef_c_2, v_coef_s_1, v_coef_s_2]) ** 2

    return u_var, v_var


def compute_variance_ad2d(config):
    basis = FullFourierBasis2d(
        num_modes=config['num_modes'],
        x_min=(config['x0'], config['y0']),
        x_max=(config['x1'], config['y1'])
    )

    gx = basis.kx() / (2 * torch.pi)  # (d)
    gy = basis.ky() / (2 * torch.pi)  # (d)
    sqrt_lam = config['alpha'] / (config['beta'] + gx**2 + gy**2) ** (config['gamma'] / 2)

    return sqrt_lam**2


@mlx.experiment
def compute_variance(config, name, group=None):
    lib = OLDatasetLibrary(config['library'])
    dataset = OLDataset(lib.dataset_path(config['split'], config['dataset_id']))
    meta = lib[config['dataset_id']]

    num_data = len(dataset)
    u_gram = torch.empty((num_data, num_data))
    v_gram = torch.empty((num_data, num_data))
    for i, (u_i, _, v_i, _) in tqdm.tqdm(enumerate(dataset), total=num_data, unit='sample'):
        for j, (u_j, _, v_j, _) in enumerate(dataset):
            u_gram[i, j] = torch.mean((u_i * u_j).sum(dim=-1))
            v_gram[i, j] = torch.mean((v_i * v_j).sum(dim=-1))

    u_vals = torch.linalg.eigvalsh(u_gram / num_data)
    v_vals = torch.linalg.eigvalsh(v_gram / num_data)

    output = {'u_vals': u_vals, 'v_vals': v_vals}

    if config['library'] == 'heat1d':
        u_vals_true, v_vals_true = compute_variance_heat1d(meta)
        output['u_vals_true'] = u_vals_true
        output['v_vals_true'] = v_vals_true
    elif config['library'] == 'wave1d':
        u_vals_true, v_vals_true = compute_variance_wave1d(meta)
        output['u_vals_true'] = u_vals_true
        output['v_vals_true'] = v_vals_true
    elif config['library'] == 'ad2d':
        u_vals_true = compute_variance_ad2d(meta)
        output['u_vals_true'] = u_vals_true

    output_file = os.path.join(
        mlx.results_dir(name, config['library']),
        f"{config['dataset_id']}-{config['split']}.pt"
    )
    torch.save(output, str(output_file))
