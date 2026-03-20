import torch
import torch.utils.data
from operatorlearning.data import OLDataset


def pd2d_to_tensor_grid(t, nx, ny):
    """
    :param t: (B, N, 2) flat tensor read from OLPDMATLAB2D output
    :param nx: number of points in the x (first) direction
    :param ny: number of points in thee y (second) direction
    :return: (B, nx, ny, 2) version of tensor conforming to
        operatorlearning.GridFunction tensor grid conventions
    """
    t = t.reshape(-1, nx, ny, 2).permute(0, 2, 1, 3)
    return torch.flip(t, (2,))


def pd2d_import_dataset(dataset_in, dataset_out, nx, ny):
    """
    Converts all tensors in the given OLPDMATLAB2D output dataset to
    the tensor grid conventions of operatorlearning.GridFunction
    :param dataset_in: Path to input .ol.h5 dataset
    :param dataset_out: Path to new output .ol.h5 dataset
    :param nx: Number of nodes in x direction
    :param ny: Number of nodes in y direction
    """
    data_in = OLDataset(dataset_in, stream_uv=False, stream_xy=False)
    u = pd2d_to_tensor_grid(data_in.u['1'], nx, ny)  # (B, nx, ny, 2)
    v = pd2d_to_tensor_grid(data_in.v['1'], nx, ny)  # (B, nx, ny, 2)
    x = pd2d_to_tensor_grid(data_in.x[1][None], nx, ny)[0]  # (nx, ny, 2)
    y = pd2d_to_tensor_grid(data_in.y[1][None], nx, ny)[0]  # (nx, ny, 2)

    disc = torch.zeros(len(u), dtype=torch.long)
    OLDataset.write(u, [x], v, [y], dataset_out, u_disc=disc, v_disc=disc)


class NormalizedOLDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        super().__init__()
        assert len(base_dataset.x) == 1, \
            'NormalizedOLDataset only supports single-representation OLDatasets'

        self.base_dataset = base_dataset

        u0, _, v0, _ = base_dataset[0]
        all_u = torch.empty((len(base_dataset), *u0.shape), device=u0.device)
        all_v = torch.empty((len(base_dataset), *v0.shape), device=v0.device)
        for i, data in base_dataset:
            all_u[i] = data[0]
            all_v[i] = data[2]

        print('Normalizing OLDataset')
        self.u_mean = torch.mean(all_u.view(-1, all_u.shape[-1]), dim=0)
        self.u_mean = self.u_mean.reshape((1,) * len(u0.shape) + (-1,))
        self.u_std = torch.mean(all_u.view(-1, all_u.shape[-1]), dim=0)
        self.u_std = self.u_std.reshape((1,) * len(u0.shape) + (-1,))
        self.v_mean = torch.mean(all_v.view(-1, all_v.shape[-1]), dim=0)
        self.v_mean = self.v_mean.reshape((1,) * len(v0.shape) + (-1,))
        self.v_std = torch.mean(all_v.view(-1, all_v.shape[-1]), dim=0)
        self.v_std = self.v_std.reshape((1,) * len(v0.shape) + (-1,))

    def __len__(self):
        return len(self.base_dataset)

    def normalize_u(self, u):
        return (u - self.u_mean) / self.u_std

    def normalize_v(self, v):
        return  (v - self.v_mean) / self.v_std

    def denormalize_u(self, u_z):
        return self.u_mean + self.u_std * u_z

    def denormalize_v(self, v_z):
        return self.v_mean + self.v_std * v_z

    def __getitem__(self, item):
        u, x, v, y = self.base_dataset[item]
        u_z = self.normalize_u(u)
        v_z = self.normalize_v(v)
        return u_z, x, v_z, y
