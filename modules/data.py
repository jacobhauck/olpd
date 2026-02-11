import numpy as np
from operatorlearning.data import OLDataset


def pd2d_to_tensor_grid(t, nx, ny):
    """
    :param t: (B, N, 2) flat tensor read from OLPDMATLAB2D output
    :param nx: number of points in the x (first) direction
    :param ny: number of points in thee y (second) direction
    :return: (B, nx, ny, 2) version of tensor conforming to
        operatorlearning.GridFunction tensor grid conventions
    """
    t = t.reshape(-1, nx, ny, 2).transpose(0, 2, 1, 3)
    return np.flip(t, 2)


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

    disc = [0] * len(u)
    OLDataset.write(u, [x], v, [y], dataset_out, u_disc=disc, v_disc=disc)
