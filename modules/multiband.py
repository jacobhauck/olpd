import torch
import mlx


class MultiLoss(torch.nn.Module):
    def __init__(self, base_loss, weights):
        super().__init__()
        self.base_loss = mlx.create_module(base_loss)
        self.register_buffer('weights', torch.tensor(weights))

    def forward(self, pred, target, x=None):
        """
        :param pred: list of (B, *shape, d_out) predicted function bands
        :param target: list of (B, *shape, d_out) target function bands
        :param x: list of (B, *shape, d_in) sample points for each band (or None,
            if loss does not require them)
        :return: tuple (band_losses, total_loss), where band_losses is a tensor
            of shape (num_steps,) of base_loss applied to each band separately
            and total_loss is band_losses averaged using weights
        """
        if x is None:
            x = [None] * len(pred)
        losses = torch.stack([
            self.base_loss(p, t, x=x_i) for p, t, x_i in zip(pred, target, x)
        ])
        return losses, torch.sum(losses * self.weights)


class MultiModel(torch.nn.Module):
    def __init__(self, models, decomposition):
        super().__init__()
        self.models = torch.nn.ModuleList([
            mlx.create_module(model) for model in models
        ])
        self.decomposition = mlx.create_module(decomposition)

    def forward(self, u, x, y):
        """
        Applies multiband model and returns either full recomposed prediction
        or separate bands
        :param u: (B, *in_shape, u_d_out) Input function values
        :param x: (B, *in_shape, u_d_in) Input function sample points
        :param y: To request recomposed prediction, (B, *out_shape, v_d_in)
            tensor of output function sample points. To request separate
            bands, a list of (B, *out_shape_i, v_d_in) tensors giving output
            sampling points for each band
        :return: (B, *out_shape, v_d_out) Output function values, if
            separate_bands is False (can only be used if all bands use the same
            size sampling grid, or if in eval mode); otherwise, returns a list
            of length num_steps of tensors with shape (B, *out_shape_i, v_d_out)
            giving the function values for each band, and points is a list of
            tensors of shape (b, *out_shape_i, v_d_in) giving the corresponding
            sample points
        """
        recompose = False
        if isinstance(y, torch.Tensor):
            recompose = True
            y = self.decomposition.get_x(y)  # get per-band sampling points
            y = [y[:, i] for i in range(self.decomposition.num_steps)]

        v = []
        for model, y_i in zip(self.models, y):
            v.append(model(u, x, x_out=y_i))

        if recompose:
            v = torch.stack(v, dim=1)
            return self.decomposition.recompose(v)
        else:
            return v


class MultibandDecomposition2d(torch.nn.Module):
    def __init__(self, low_freq_cutoff, dilation, num_steps):
        super().__init__()
        self.low_freq_cutoff = mlx.create_module(low_freq_cutoff)
        self.dilation = dilation
        self.num_steps = num_steps

    def get_x(self, x):
        """
        :param x: (B, N1, N2, 2) base function (band 0) sample points
        :return: (B, num_steps, N1, N2) sample points for each band
        """
        y = torch.empty(
            (x.shape[0], self.num_steps, *x.shape[1:]),
            dtype=x.dtype, device=x.device
        )
        x_cur = x
        for step in range(self.num_steps):
            y[:, step] = x_cur
            x_cur = self.dilation * x_cur

        return y

    def forward(self, u, x):
        """
        :param u: (B, N1, N2, d_out)
        :param x: (B, N1, N2, 2)
        :return: tuple v, y, where v has shape (B, num_steps, N1, N2, d_out)
            and y has shape (B, num_steps, N1, N2, 2), which give the
            multiband decomposition of u
        """
        start_x_max = self.low_freq_cutoff.x_max.clone()
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
            x_cur = self.dilation * x_cur
            self.low_freq_cutoff.set_x_max(self.dilation * self.low_freq_cutoff.x_max)

        self.low_freq_cutoff.set_x_max(start_x_max)

        return v, y

    def recompose(self, v):
        """
        :param v: (B, num_steps, N1, N2, d_out) Band projection sample values
        :return: (B, num_steps, N1, N2, d_out) Reconstructed function sample values
        """
        weights = torch.pow(self.dilation, -2 * torch.arange(self.num_steps))
        return torch.einsum('BnXYD,n->BXYD', v, weights.to(v.device))


class SingleBandTransform(torch.nn.Module):
    def __init__(self, decomposition, band_index):
        super().__init__()
        self.decomposition = mlx.create_module(decomposition)
        self.band_index = band_index

    def forward(self, u, x):
        u_d, x_d = self.decomposition(u, x)
        return u_d[:, self.band_index], x_d[:, self.band_index]
