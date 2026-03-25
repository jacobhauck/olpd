import mlx
import torch


def fftfreqn(shape):
    """
    Generates FFT sample frequencies for a signal of given shape
    :param shape: (s_1, s_2, ..., s_n) tuple of sizes
    :return: (s_1, s_2, ..., s_n, n) array of sample frequencies
    """
    freq1d = [torch.fft.fftfreq(s) for s in shape]
    return torch.stack(torch.meshgrid(*freq1d, indexing='ij'), dim=-1)


def rfftfreqn(shape):
    """
    Generates real FFT sample frequencies for a signal of given shape
    :param shape: (s_1, s_2, ..., s_n) tuple of sizes
    :return: (s_1, s_2, ..., s_n, n) array of sample frequencies
    """
    freq1d = [torch.fft.fftfreq(s) for s in shape[:-1]]
    freq1d.append(torch.fft.rfftfreq(shape[-1]))
    return torch.stack(torch.meshgrid(*freq1d, indexing='ij'), dim=-1)


class SmoothedLoss(torch.nn.Module):
    def __init__(self, base_loss, smoothing_operator):
        """
        :param base_loss: Config for base loss function that evaluates loss
            after smoothing
        :param smoothing_operator: Config for a module that smoothes the target
            function
        """
        super().__init__()
        self.base_loss = mlx.create_module(base_loss)
        self.smoothing_operator = mlx.create_module(smoothing_operator)

    def __repr__(self):
        return f'SmoothedLoss(base={self.base_loss}, op={self.smoothing_operator})'

    def forward(self, prediction, target):
        """
        :param prediction: (B, *shape, v_d_out) predicted function values
        :param target: (B, *shape, v_d_out) target function values
        :return: Relative L^2 error between prediction and target restricted to
            the low-frequency ball determined by self.smoothing_operator
        """
        return self.base_loss(prediction, self.smoothing_operator(target))


class GaussianSmoothing2d(torch.nn.Module):
    def __init__(self, kernel_sigma, x_min, x_max, n, extend='constant', adaptive=False):
        """
        Applies convolution with a Gaussian kernel to smooth the 2D input
        function. Function must be sampled at centers of uniform grid of squares
        on a rectangular domain (so we can use separable calls to
        torch.nn.functional.conv2d to efficiently perform the smoothing).

        :param kernel_sigma: Sigma of the Gaussian kernel
        :param x_min: Minimum point of the rectangular domain (list of 2 floats)
        :param x_max: Maximum point of the rectangular domain (list of 2 floats)
        :param n: Number of cells in each direction (list of 2 ints)
        :param extend: How to extend the input function to the whole plane to
            make the convolution work. Choose from 'zero', 'reflect', or
            'periodic', 'constant'. Default = 'constant'.
        :param adaptive: Whether the operator should adaptively rebuild the
            kernel in response to a new resolution. Default = False
        """
        super().__init__()
        self.kernel_sigma = kernel_sigma
        self.register_buffer('x_min', torch.tensor(list(map(float, x_min))))
        self.register_buffer('x_max', torch.tensor(list(map(float, x_max))))
        self.register_buffer('n', torch.tensor(list(map(int, n)), dtype=torch.long))
        self.kernel_radius, w0, w1 = self._build_weights()
        self.register_buffer('weight0', w0)
        self.register_buffer('weight1', w1)
        self.extend = extend
        self.adaptive = adaptive

    def set_x_max(self, x_max):
        self.x_max[:] = x_max
        self.kernel_radius, w0, w1 = self._build_weights()
        self.weight0 = w0
        self.weight1 = w1

    def set_n(self, n):
        self.n[:] = n
        self.kernel_radius, w0, w1 = self._build_weights()
        self.weight0 = w0
        self.weight1 = w1

    def _build_weights(self):
        # Calculate cell size
        dx = (self.x_max - self.x_min) / self.n.to(torch.float)

        # 4 sigma should be enough for numerical accuracy
        kernel_radius = torch.ceil(4 * self.kernel_sigma / dx).to(torch.long)

        # L^1 normalization for the Gaussian (in 1D, since we are using
        # separable convolutions with 1D Gaussians to implement the 2D
        # convolution)
        a = (2 * torch.pi) ** 0.5 * self.kernel_sigma

        x0 = dx[0] * torch.arange(
            -kernel_radius[0], kernel_radius[0] + 1,
            dtype=torch.float, device=self.n.device
        )
        # Pre-multiply quadrature weight dx[0] (for centered Riemann sum approximation)
        w0 = torch.exp(-x0**2 / (2 * self.kernel_sigma**2)) / a * dx[0]
        w0 = w0.reshape(1, 1, -1, 1)

        x1 = dx[1] * torch.arange(
            -kernel_radius[1], kernel_radius[1] + 1,
            dtype=torch.float, device=self.n.device
        )
        # Pre-multiply quadrature weight dx[0] (for centered Riemann sum approximation)
        w1 = torch.exp(-x1**2 / (2 * self.kernel_sigma**2)) / a * dx[1]
        w1 = w1.reshape(1, 1, 1, -1)

        return tuple(map(int, kernel_radius)), w0, w1

    def forward(self, u):
        """
        :param u: (B, N0, N1, v_d_out) function values on the grid
        :return: (B, N0, N1, v_d_out) smoothed function values on the grid
        """
        batch_size, n0, n1, v_d_out = u.shape

        if self.n[0] != n0 or self.n[1] != n1:
            self.n[0] = n0
            self.n[1] = n1
            self.kernel_radius, w0, w1 = self._build_weights()
            self.weight0 = w0
            self.weight1 = w1

        # Move output components to batch dimension to apply convolution
        # component-wise, and add dummy channel dimension for compliance with
        # torch.nn.functional.conv2d interface
        u = torch.permute(u, (0, 3, 1, 2)).reshape(-1, 1, *u.shape[1:3])
        # (B*v_d_out, 1, N0, N1)

        if self.extend == 'zero':
            u0 = torch.nn.functional.conv2d(u, self.weight0, padding=self.kernel_radius[0])
            u1 = torch.nn.functional.conv2d(u0, self.weight1, padding=self.kernel_radius[1])
        else:
            if self.extend == 'reflect':
                mode = 'reflect'
            elif self.extend == 'constant':
                mode = 'replicate'
            elif self.extend == 'periodic':
                mode = 'circular'
            else:
                raise ValueError('Invalid extend mode for smoothing')

            pad = (
                self.kernel_radius[1], self.kernel_radius[1],
                self.kernel_radius[0], self.kernel_radius[0]
            )
            u = torch.nn.functional.pad(u, pad, mode=mode)
            u0 = torch.nn.functional.conv2d(u, self.weight0)
            u1 = torch.nn.functional.conv2d(u0, self.weight1)

        # Go back to standard shape/ordering
        return torch.permute(u1.reshape(batch_size, v_d_out, *u1.shape[2:]), (0, 2, 3, 1))
