import mlx
import torch
from operatorlearning.data import OLDataset, OLDatasetLibrary


@mlx.experiment
def compute_variance(config, name, group=None):
    lib = OLDatasetLibrary('nonlinear')
    dataset = OLDataset(lib.dataset_path(config['split'], config['dataset_id']))
    u, x, v, y = dataset[0]
    n_features = u.flatten().shape[0]
    size = len(dataset)
    data_mat_u = torch.empty((size, n_features))
    data_mat_v = torch.empty((size, n_features))

    for i, (u, x, v, y) in enumerate(dataset):
        data_mat_u[i] = u.flatten()
        data_mat_v[i] = v.flatten()

    _, lam_u, _ = torch.linalg.svd(data_mat_u)
    _, lam_v, _ = torch.linalg.svd(data_mat_v)

    torch.save(lam_u, 'results/nonlinear/lam_u.pt')
    torch.save(lam_v, 'results/nonlinear/lam_v.pt')