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


class SmoothRelativeL2Loss(torch.nn.Module):
    def __init__(self, bandwidth, x_min, x_max, squared=True):
        """
        :param bandwidth: Smoothing bandwidth (i.e., frequency cutoff used to
            smooth functions before computing L^2 error)
        :param x_min: list of d numbers; lower point of domain
        :param x_max: list of d numbers; upper point of domain
        :param squared: Whether to return squared error or not
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.x_min = torch.tensor(x_min, dtype=torch.float)
        self.x_max = torch.tensor(x_max, dtype=torch.float)
        self.squared = squared

    def forward(self, prediction, target):
        """
        Functions must be sampled on a regular grid with minimum point
        self.x_min and maximum point self.x_max

        :param prediction: (B, *shape, v_d_out) predicted function values
        :param target: (B, *shape, v_d_out) target function values
        :return: Relative L^2 error between prediction and target restricted to
            the low-frequency ball of radius `self.bandwidth`
        """
        d = len(self.x_min)
        assert len(prediction.shape) == d + 2, 'prediction has invalid shape'
        assert len(target.shape) == d + 2, 'target has invalid shape'

        f = rfftfreqn(target.shape[1:-1]) / (self.x_max - self.x_min)
        f *= torch.tensor(target.shape[1:-1])
        mask = (f < self.bandwidth).to(target.dtype).to(target.device)

        dims = tuple(range(-d - 1, -1))
        prediction_fft = torch.fft.rfftn(prediction, dim=dims)
        target_fft = torch.fft.rfftn(target, dim=dims)

        error = torch.mean(((prediction_fft - target_fft) * mask).abs() ** 2, dim=dims)
        magnitude = torch.mean((target_fft * mask).abs() ** 2, dim=dims)
        # (B, d) each

        if self.squared:
            return torch.mean(error.sum(dim=1) / magnitude.sum(dim=1))
        else:
            return torch.mean((error.sum(dim=1) / magnitude.sum(dim=1)) ** .5)
