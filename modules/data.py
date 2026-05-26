import torch
import torch.utils.data
from operatorlearning.data import OLDataset


def pd2d_to_tensor_grid(t, nx, ny):
    """
    :param t: (B, N, d) flat tensor read from OLPDMATLAB2D output
    :param nx: number of points in the x (first) direction
    :param ny: number of points in thee y (second) direction
    :return: (B, nx, ny, d) version of tensor conforming to
        operatorlearning.GridFunction tensor grid conventions
    """
    d = t.shape[-1]
    t = t.reshape(-1, ny, nx, d).permute(0, 2, 1, 3)
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


def pd2d_subsample_dataset(dataset_in, dataset_out, nx, ny):
    """
    Subsamples the given dataset to a grid with the requested dimensions
    (which must divide the source dimensions)

    :param dataset_in: Name of input dataset file
    :param dataset_out: Name of file to save subsampled dataset to
    :param nx: Number of points in x direction
    :param ny: Number of points in y direction
    """
    data_in = OLDataset(dataset_in, stream_uv=False, stream_xy=False)
    nx_in, ny_in = data_in.u['0'].shape[1:3]
    assert nx_in % nx == 0 and ny_in % ny == 0, \
        'Subsampling resolution must divide original'
    x_step = nx_in // nx
    y_step = ny_in // ny

    u = data_in.u['0'][:, ::x_step, ::y_step]  # (B, nx, ny, 2)
    v = data_in.v['0'][:, ::x_step, ::y_step]  # (B, nx, ny, 2)
    x = data_in.x[0][::x_step, ::y_step]  # (nx, ny, 2)
    y = data_in.y[0][::x_step, ::y_step]  # (nx, ny, 2)

    disc = torch.zeros(len(u), dtype=torch.long)
    OLDataset.write(u, [x], v, [y], dataset_out, u_disc=disc, v_disc=disc)


def crack1_import_dataset(dataset_in, dataset_out, nx, ny):
    """
    Converts all tensors in the given OLPDMATLAB2D output dataset to
    the tensor grid conventions of operatorlearning.GridFunction
    :param dataset_in: Path to input .ol.h5 dataset
    :param dataset_out: Path to new output .ol.h5 dataset
    :param nx: Number of nodes in x direction
    :param ny: Number of nodes in y direction
    """
    data_in = OLDataset(dataset_in, stream_uv=False, stream_xy=False)
    v = pd2d_to_tensor_grid(data_in.v['1'], nx, ny)  # (B, nx, ny, 1)
    y = pd2d_to_tensor_grid(data_in.y[1][None], nx, ny)[0]  # (nx, ny, 2)

    disc = torch.zeros(len(v), dtype=torch.long)
    OLDataset.write(data_in.u['1'], [data_in.x[1]], v, [y], dataset_out, u_disc=disc, v_disc=disc)


def crack1_subsample_dataset(dataset_in, dataset_out, nbx, nx, ny):
    """
    Subsamples the given dataset to a grid with the requested dimensions
    (which must divide the source dimensions for nx and ny, and must generate
    a valid subsampling for nbx (which uses a closed sampling rule))

    :param dataset_in: Name of input dataset file
    :param dataset_out: Name of file to save subsampled dataset to
    :param nbx: Number of sampling points on the boundary (in x direction)
    :param nx: Number of points in x direction
    :param ny: Number of points in y direction
    """
    data_in = OLDataset(dataset_in, stream_uv=False, stream_xy=False)
    nbx_in = data_in.u['0'].shape[1]
    assert (nbx_in - 1) % (nbx - 1) == 0, \
        'nbx must be valid number of boundary subsampling points'
    bx_step = (nbx_in - 1) // (nbx - 1)

    nx_in, ny_in = data_in.v['0'].shape[1:3]
    assert nx_in % nx == 0 and ny_in % ny == 0, \
        'Subsampling resolution must divide original'
    x_step = nx_in // nx
    y_step = ny_in // ny

    u = data_in.u['0'][:, ::bx_step]  # (B, nbx, 2)
    v = data_in.v['0'][:, ::x_step, ::y_step]  # (B, nx, ny, 1)
    x = data_in.x[0][::bx_step]  # (nbx, 1)
    y = data_in.y[0][::x_step, ::y_step]  # (nx, ny, 2)

    disc = torch.zeros(len(u), dtype=torch.long)
    OLDataset.write(u, [x], v, [y], dataset_out, u_disc=disc, v_disc=disc)


def crack1_crop_dataset(dataset_in, dataset_out, i_x_min, i_x_max, i_y_min, i_y_max):
    """
    Crops outputs in a crack1 dataset and saves the result in a new dataset.
    :param dataset_in: Path to input dataset
    :param dataset_out: Path to output dataset
    :param i_x_min: Index of x cell from which to start crop (inclusive)
    :param i_x_max: Index of x cell at which to stop crop (exclusive)
    :param i_y_min: Index of y cell from which to start crop (inclusive)
    :param i_y_max: Index of y cell at which to stop crop (exclusive)
    """
    data_in = OLDataset(dataset_in, stream_uv=False, stream_xy=False)
    u = data_in.u['0']
    x = data_in.x[0]

    v = data_in.v['0'][:, i_x_min:i_x_max, i_y_min:i_y_max]  # (B, nx', ny', 1)
    y = data_in.y[0][i_x_min:i_x_max, i_y_min:i_y_max]  # (nx', ny', 2)

    disc = torch.zeros(len(v), dtype=torch.long)
    OLDataset.write(u, [x], v, [y], dataset_out, u_disc=disc, v_disc=disc)


class NormalizedOLDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            base_dataset,
            u_mean=None,
            u_std=None,
            v_mean=None,
            v_std=None
    ):
        super().__init__()
        assert len(base_dataset.x) == 1, \
            'NormalizedOLDataset only supports single-representation OLDatasets'

        self.base_dataset = base_dataset

        u0, _, v0, _ = base_dataset[0]
        all_u = torch.empty((len(base_dataset), *u0.shape), device=u0.device)
        all_v = torch.empty((len(base_dataset), *v0.shape), device=v0.device)
        for i, data in enumerate(base_dataset):
            all_u[i] = data[0]
            all_v[i] = data[2]

        if u_mean is None:
            print('Normalizing OLDataset')
            self.u_mean = torch.mean(all_u.view(-1, all_u.shape[-1]), dim=0)
            self.u_mean = self.u_mean.reshape((1,) * (len(u0.shape) - 1) + (-1,))
            print('Calculated u_mean:', self.u_mean)
        else:
            self.u_mean = u_mean.clone()

        if u_std is None:
            self.u_std = torch.std(all_u.view(-1, all_u.shape[-1]), dim=0)
            self.u_std = self.u_std.reshape((1,) * (len(u0.shape) - 1) + (-1,))
            print('Calculated u_std:', self.u_std)
        else:
            self.u_std = u_std.clone()

        if v_mean is None:
            self.v_mean = torch.mean(all_v.view(-1, all_v.shape[-1]), dim=0)
            self.v_mean = self.v_mean.reshape((1,) * (len(v0.shape) - 1) + (-1,))
            print('Calculated v_mean:', self.v_mean)
        else:
            self.v_mean = v_mean.clone()

        if v_std is None:
            self.v_std = torch.std(all_v.view(-1, all_v.shape[-1]), dim=0)
            self.v_std = self.v_std.reshape((1,) * (len(v0.shape) - 1) + (-1,))
            print('Calculated v_std:', self.v_std)
        else:
            self.v_std = v_std.clone()

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
