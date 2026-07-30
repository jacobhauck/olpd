import mlx
import torch
import tqdm
from operatorlearning.data import OLDataset, OLDatasetLibrary


@mlx.experiment
def generate(config, name, group=None):
    if 'seed' in config:
        torch.random.manual_seed(config['seed'])
    n = torch.arange(0, config['num_modes'] + 1).to(torch.float)  # (num_modes + 1,)
    x = torch.linspace(config['x0'], config['x1'], config['mesh_size'])  # (mesh_size,)
    x += (x[1] - x[0]) / 2
    size = config['x1'] - config['x0']
    sq_lam = config['alpha'] / (config['beta'] + (n*torch.pi/size)**2) ** (config['gamma'] / 2)
    # (num_modes + 1,)

    k = config['k']
    t = config['t_final']
    pi = torch.pi

    basis_c = sq_lam[None] * torch.cos(n[None] * x[:, None] * 2 * pi / size) * (2/size)**.5
    basis_c[:, 0] /= 2**.5
    # (mesh_size, num_modes + 1)
    basis_s = sq_lam[None, 1:] * torch.sin(n[None, 1:] * x[:, None] * 2 * pi / size) * (2/size)**.5
    # (mesh_size, num_modes)

    data_lib = OLDatasetLibrary('heat1d')
    dataset_id = data_lib.create_dataset(
        k=config['k'],
        x0=config['x0'],
        x1=config['x1'],
        t_final=config['t_final'],
        mesh_size=config['mesh_size'],
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        num_modes=config['num_modes']
    )

    try:
        for split_name, split_size in config['splits'].items():
            print(f'Generating split {split_name}')
            all_u = []
            all_v = []

            for _ in tqdm.tqdm(range(split_size)):
                coef_c_1 = torch.randn((1, len(n)))  # (1, num_modes + 1)
                coef_s_1 = torch.randn((1, len(n) - 1))  # (1, num_modes)
                coef_c_2 = torch.randn((1, len(n)))  # (1, num_modes + 1)
                coef_s_2 = torch.randn((1, len(n) - 1))  # (1, num_modes)
                u_1 = (coef_c_1 * basis_c).sum(dim=1) + (coef_s_1 * basis_s).sum(dim=1)  # (mesh_size,)
                u_2 = (coef_c_2 * basis_c).sum(dim=1) + (coef_s_2 * basis_s).sum(dim=1)  # (mesh_size,)
                all_u.append(torch.stack([u_1, u_2], dim=1))  # (mesh_size, 2)

                v_coef_c_1 = coef_c_1 * torch.exp(-k * 4 * n[None, :]**2 * pi**2 / size**2 * t)
                # (1, num_modes + 1)
                v_coef_c_2 = coef_c_2 * torch.exp(-k * 4 * n[None, :]**2 * pi**2 / size**2 * t)
                # (1, num_modes + 1)
                v_coef_s_1 = coef_s_1 * torch.exp(-k * 4 * n[None, 1:]**2 * pi**2 / size**2 * t)
                # (1, num_modes + 1)
                v_coef_s_2 = coef_s_2 * torch.exp(-k * 4 * n[None, 1:]**2 * pi**2 / size**2 * t)
                # (1, num_modes + 1)

                v_1 = (v_coef_c_1 * basis_c).sum(dim=1) + (v_coef_s_1 * basis_s).sum(dim=1)  # (mesh_size,)
                v_2 = (v_coef_c_2 * basis_c).sum(dim=1) + (v_coef_s_2 * basis_s).sum(dim=1)  # (mesh_size,)
                all_v.append(torch.stack([v_1, v_2], dim=1))  # (mesh_size, 2)

            output_file = data_lib.dataset_path(split_name, dataset_id)
            print(f'Saving dataset split at {output_file}')
            OLDataset.write(
                all_u, [x[:, None]], all_v, [x[:, None]],
                file_name=output_file,
                u_disc=torch.zeros(split_size, dtype=torch.long),
                v_disc=torch.zeros(split_size, dtype=torch.long)
            )

    except Exception as e:
        print('Aborting dataset generation due to an error')
        data_lib.delete_dataset(dataset_id)
        raise e