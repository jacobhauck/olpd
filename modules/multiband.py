import torch
import mlx


class MultibandDecomposition2d(torch.nn.Module):
    def __init__(self, low_freq_cutoff, dilation, num_steps):
        super().__init__()
        self.low_freq_cutoff = mlx.create_module(low_freq_cutoff)
        self.dilation = dilation
        self.num_steps = num_steps

    def forward(self, u, x):
        """
        :param u: (B, N1, N2, d_out)
        :param x: (B, N1, N2, d_in)
        :return: tuple v, y, where v has shape (B, num_steps, N1, N2, d_out)
            and y has shape (B, num_steps, N1, N2, d_in), which give the
            multiband decomposition of u
        """
        start_x_max = self.low_freq_cutoff.x_max
        v = torch.empty(
            (u.shape[0], self.num_steps, *u.shape[1:]),
            dtype=u.dtype, device=u.device
        )
        y = torch.empty(
            (v.shape[0], self.num_steps, *x.shape[1:]),
            dtype=x.dtype, device=x.device
        )

        u_cur = u
        x_cur = x
        for step in range(self.num_steps):
            u_smooth = self.low_freq_cutoff(u_cur)
            v[:, step] = u_smooth
            y[:, step] = x_cur

            u_cur = self.dilation**2 * (u_cur - u_smooth)
            x_cur = self.dilation * x
            self.low_freq_cutoff.set_x_max(self.dilation * self.low_freq_cutoff.x_max)

        self.low_freq_cutoff.set_x_max(start_x_max)

        return v, y


class SingleBandTransform(torch.nn.Module):
    def __init__(self, decomposition, band_index):
        super().__init__()
        self.decomposition = mlx.create_module(decomposition)
        self.band_index = band_index

    def forward(self, u, x):
        u_d, x_d = self.decomposition(u, x)
        return u_d[:, self.band_index], x_d[:, self.band_index]
