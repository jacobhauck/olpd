import mlx
import torch
import matplotlib.pyplot as plt
from operatorlearning import GridFunction


def falling(x, p):
    result = torch.ones_like(x)
    for j in range(p):
        result = result * (x - j)
    return result


class SmoothBump1d(torch.nn.Module):
    def __init__(self, n, a, height):
        super().__init__()
        self.n = n
        self.a = a
        self.height = height
        if n > 0:
            self.register_buffer('coef', self.calc_coefficients())

    def forward(self, x):
        if self.n > 0:
            out = (torch.abs(x) > self.a)
            left = (~out) & (x < 0)
            right = ~(out | left)
            result = torch.zeros_like(x)
            result[left] = self.eval_left(x[left])
            result[right] = self.eval_left(-x[right])
        else:
            # noinspection PyTypeChecker
            result = torch.where(
                torch.abs(x) > self.a/2,
                torch.zeros_like(x),
                torch.full_like(x, self.height)
            )

        return result

    def eval_left(self, x):
        cur_power = 1.0
        result = torch.zeros_like(x)
        d = x + self.a
        for j in range(self.n):
            result = result + self.coef[j] * cur_power
            cur_power = cur_power * d

        return result * cur_power

    def calc_coefficients(self):
        matrix = torch.empty((self.n, self.n))
        n_vals = torch.arange(self.n, 2 * self.n)
        for row in range(self.n):
            matrix[row, :] = falling(n_vals, row)

        rhs = torch.cat([torch.ones(1), torch.zeros(self.n-1)])
        rhs *= self.height / (self.a ** self.n)
        c_scaled = torch.linalg.solve(matrix, rhs)
        return c_scaled * (self.a ** (-torch.arange(self.n)))


class SmoothBump2d(torch.nn.Module):
    def __init__(self, n, a, height):
        super().__init__()
        self.profile = SmoothBump1d(n, a, height)

    def forward(self, x):
        x_, y_ = x[..., 0], x[..., 1]
        r_ = torch.sqrt(x_**2 + y_**2)
        return self.profile(r_)[..., None]


@mlx.experiment
def run_experiment(config, name, group=None):
    fig, axes = plt.subplots(1, config['max_n'] + 1, figsize=((config['max_n'] + 1) * 3, 4))
    for n in range(config['max_n'] + 1):
        bump = SmoothBump2d(n, config['width']/2, config['height'])
        if n == 2:
            print(bump.profile.coef)
        x = GridFunction.uniform_x(torch.tensor(config['min_point']), torch.tensor(config['max_point']), 200)
        axes[n].set_axis_off()
        axes[n].set_title(str(n))
        axes[n].imshow(bump(x)[..., 0].cpu())
    plt.show()
