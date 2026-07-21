import mlx
import torch
import tqdm
from operatorlearning.data import OLDataset


@mlx.experiment
def generate(config, name, group=None):
    n = torch.arange(1, config['num_modes'] + 1).to(torch.float)
    x = torch.linspace(config['x0'], config['x1'], config['mesh_size'])
    x += (x[1] - x[0]) / 2
    size = config['x1'] - config['x0']
    sq_lam = config['alpha'] / (config['beta'] + (n*torch.pi/size)**2) ** (config['gamma'] / 2)
    basis = sq_lam[None] * torch.cos(n[None] * x[:, None] * torch.pi / size)
    basis_v = basis * torch.exp(-n[None]**2 * torch.pi**2 * config['k'] * config['t_final'] / size**2)

    for split_name, split_size in config['splits'].items():
        print(f'Generating split {split_name}')
        all_u = []
        all_v = []

        for _ in tqdm.tqdm(range(split_size)):
            coef = torch.randn((1, len(n)))
            all_u.append((coef * basis).sum(dim=1)[:, None])
            all_v.append((coef * basis_v).sum(dim=1)[:, None])

        output_file = 'data/heat1d/' + split_name + str(config['dataset_id']) + '.ol.h5'
        print(f'Saving dataset split at {output_file}')
        OLDataset.write(
            all_u, [x[:, None]], all_v, [x[:, None]],
            file_name=output_file,
            u_disc=torch.zeros(split_size, dtype=torch.long),
            v_disc=torch.zeros(split_size, dtype=torch.long)
        )
