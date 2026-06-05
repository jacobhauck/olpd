import mlx
from operatorlearning.data import OLDataset
from operatorlearning.modules import PCANet, FunctionalL2Loss
import torch
import matplotlib.pyplot as plt


@mlx.experiment
def run_experiment(config, name, group=None):
    dataset = OLDataset(config['dataset'])
    test_dataset = OLDataset(config['test_dataset'])
    pcanet = PCANet(
        u_num_modes=config['u_max_modes'],
        u_sample_shape=dataset[0][0].shape,
        v_num_modes=config['v_max_modes'],
        v_sample_shape=dataset[0][2].shape,
        approximator={
            'name': 'Linear',
            'in_features': config['u_max_modes'],
            'out_features': config['v_max_modes']
        }
    )

    print('Fitting PCA bases')
    pcanet.fit_pca(map(lambda sample: (sample[0], sample[2]), dataset), dataset[0][1], dataset[0][3])

    print('Calculating errors')
    rel_l2 = FunctionalL2Loss(relative=True, squared=False)
    u_errors = []
    v_errors = []
    u_proj_mat = torch.empty((len(dataset), config['u_max_modes']))
    v_proj_mat = torch.empty((len(dataset), config['v_max_modes']))
    for i, (u, _, v, _) in enumerate(dataset):
        u_proj_mat[i, :] = pcanet.encoder(u[None])[0]
        u_pred = pcanet.decoder(u_proj_mat[i:i+1])[0]
        v_proj_mat[i, :] = pcanet.projector(v[None])[0]
        v_pred = pcanet.reconstructor(v_proj_mat[i:i+1])[0]
        u_errors.append(rel_l2(u_pred, u))
        v_errors.append(rel_l2(v_pred, v))

    u_errors = torch.tensor(u_errors)
    v_errors = torch.tensor(v_errors)

    print('Avg u error', u_errors.mean())
    print('Avg v error', v_errors.mean())

    weight = torch.linalg.lstsq(u_proj_mat, v_proj_mat).solution

    v_pred_errors = []
    for u, _, v, _ in test_dataset:
        u_proj = pcanet.encoder(u[None])
        v_proj_pred = u_proj @ weight
        v_pred = pcanet.reconstructor(v_proj_pred)[0]
        v_pred_errors.append(rel_l2(v_pred, v))

    v_pred_errors = torch.tensor(v_pred_errors)

    print('Avg pred v error', v_pred_errors.mean())
