import torch
import mlx
from operatorlearning import GridFunction


class TestAdaptiveSmoothingExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        smoothing_op = mlx.create_module(config['smoothing_op'])

        x = GridFunction.uniform_x(smoothing_op.x_min, smoothing_op.x_max, smoothing_op.n)
        f = torch.sin(torch.pi * x[..., 0]) * torch.cos(1.5 * torch.pi * x[..., 1])
        f_sub = f[::5, ::4]

        f = f[..., None]
        f_sub = f_sub[..., None]
        f_smooth = smoothing_op(f[None])[0]
        f_sub_smooth = smoothing_op(f_sub[None])[0]
        print((f_smooth[::5, ::4] - f_sub_smooth).abs().mean() / f_smooth.abs().mean())
