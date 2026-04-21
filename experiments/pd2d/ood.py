import mlx
from operatorlearning import OLDataset

from .pd2d import PD2DTrainer
import torch
import matplotlib.pyplot as plt
import os


class GRFBasis(torch.nn.Module):
    def __init__(self, num_modes, alpha, beta, gamma, x0, x1, y0, y1, integrator):
        super().__init__()
        self.num_modes = num_modes
        g_x, g_y = torch.meshgrid(
            torch.arange(-num_modes, num_modes + 1),
            torch.arange(-num_modes, num_modes + 1),
            indexing='ij'
        )  # each (2m+1, 2m+1)
        g_x = g_x.reshape(-1)  # ((2m+1)^2,)
        g_y = g_y.reshape(-1)  # ((2m+1)^2,)
        self.register_buffer('g_x', g_x)
        self.register_buffer('g_y', g_y)

        self.register_buffer('nonzero', (self.g_x != 0) & (self.g_y != 0))
        # ((2m+1)^2,)
        self.register_buffer('a', alpha / (beta + self.g_x**2 + self.g_y**2)**(gamma/2))
        # ((2m+1)^2,)
        f = ((x1 - x0) * (y1 - y0) / 2) ** .5
        s_var = (f * self.a[self.nonzero]) ** 2  # ((2m+1)^2-1,)
        c_var = (f * self.a) ** 2  # ((2m+1)^2,)
        self.register_buffer('cov', torch.diag(torch.cat([s_var, c_var])))
        # (2(2m+1)^2-1, 2(2m+1)^2-1)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.integrator = mlx.create_module(integrator)

    def coefficients(self, u, x):
        """
        :param u: (*shape, 2) function to project onto subspace
        :param x: (*shape, 2) values at which u is sampled
        :return: (n) projection coefficients on the basis
        """
        basis = self.evaluate(x)  # (n, *shape, 1)
        c = self.integrator(basis * u[None], x[None].expand(basis.shape[0], *x.shape))
        return c[:, 0]

    def project(self, u, x):
        """
        :param u: (*shape, 2) function to project onto subspace
        :param x: (*shape, 2) values at which u is sampled
        :return: (*shape, 2) projection of u onto space spanned by the basis
        """
        basis = self.evaluate(x)  # (n, *shape, 2)
        c = self.integrator(basis * u[None], x[None].expand(basis.shape[0], *x.shape))
        # (n, 2)
        return torch.einsum('nd,n...d->...d', c, basis)  # (*shape, 2)

    def evaluate(self, x):
        """
        :param x: (*shape, 2) sample points at which to evaluate basis
        :return: (n, *shape, 1) basis functions evaluated at sample points
        """
        x_, y_ = x.reshape(-1, 2).T  # each (prod(shape),)
        px = self.g_x[:, None] * (2*torch.pi * (x_[None] - self.x0) / (self.x1 - self.x0))
        py = self.g_y[:, None] * (2*torch.pi * (y_[None] - self.y0) / (self.y1 - self.y0))
        phase = px + py
        # each ((2m+1)^2, prod(shape))
        f = ((self.x1 - self.x0) * (self.y1 - self.y0) / 2) ** .5
        s = torch.sin(phase) / f
        s = s[self.nonzero]  # ((2m+1)^2-1, prod(shape))
        c = torch.cos(phase) / f  # ((2m+1)^2, prod(shape))

        return torch.cat([s, c], dim=0).reshape(-1, *x.shape[:-1], 1)


class OODExperiment(mlx.Experiment):
    def run(self, config, name, group=None):
        run = mlx.load_run(config['run_id'])
        run.config['device'] = config['device']
        trainer = PD2DTrainer(run.config, run)
        grf = GRFBasis(**config['grf_params']).to(config['device'])

        dataset = OLDataset(config['dataset'])

        u, x, v, y = dataset[0]
        u, x = u.to(config['device']), x.to(config['device'])
        v, y = v.to(config['device']), y.to(config['device'])
        encoder_basis = trainer.model.encoder_net(x[None])[0]  # (*shape, p, 2)
        dims = (len(x.shape) - 1, *range(0, len(x.shape) - 1), -1)
        encoder_basis = torch.permute(encoder_basis, dims)  # (p, *shape, 2)
        grf_basis = grf.evaluate(x)  # (n, *shape, 2)

        n = len(grf_basis)
        p = len(encoder_basis)

        p_mat = torch.empty((p, n))  # (p, n)
        x_batch = torch.tile(x[None], (p, 1, 1, 1))  # (p, *shape, 2)
        for i in range(n):
            p_mat[:, i] = grf.integrator(encoder_basis * grf_basis[i:i+1], x_batch)[:, 0]

        prod = torch.einsum('...d,p...d->p...', encoder_basis, u)  # (p, *shape, 2)
        z0 = trainer.model.integrator(prod[None, ..., None], x[None])[0, :, 0]  # (p)

        u_mat, d, v_mat_t = torch.linalg.svd(p_mat)  # (p, p), (p), (n, n)
        t = torch.diagonal(1/d) @ u_mat.T @ z0  # (p)
        sig_prime = v_mat_t @ grf.cov @ v_mat_t.T
        sig21 = sig_prime[p:, :p]  # (n - p, p)
        sig11 = sig_prime[:p, :p]  # (p, p)
        mu_bar = sig21 @ torch.linalg.inv(sig11) @ t  # (n - p)
        mu_prob =  torch.cat([t, mu_bar])  # (n)
        a_prob = v_mat_t @ mu_prob  # (n)
        u_prob = torch.einsum('n,n...d->...d', a_prob, grf_basis)  # (*shape, 2)

        u_proj = grf_basis.project(u, x)  # (*shape, 2)
        prob_prod = torch.einsum('...d,p...d->p...', encoder_basis, u_prob)  # (p, *shape, 2)
        prob_z0 = trainer.model.integrator(prob_prod[None, ..., None], x[None])[0, :, 0]  # (p)

        proj_prod = torch.einsum('...d,p...d->p...', encoder_basis, u_proj)  # (p, *shape, 2)
        proj_z0 = trainer.model.integrator(proj_prod[None, ..., None], x[None])[0, :, 0]  # (p)

        print('Original')
        print(z0)
        print('Most likely')
        print(prob_z0)
        print('Projected')
        print(proj_z0)

        v_min = float(u.min())
        v_max = float(u.max())

        with torch.no_grad():
            v_prob = trainer.model(u_prob[None], x[None], y[None])[0]
            v_proj = trainer.model(u_proj[None], x[None], y[None])[0]

        im_kwargs = {
            'vmin': v_min,
            'vmax': v_max,
            'cmap': 'seismic',
            'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
            'origin': 'lower'
        }

        fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 8))
        axes[0][0].imshow(u[:, :, 0].T.cpu(), **im_kwargs)
        axes[0][0].set_title(f'Original $x$ displacement')
        axes[0][0].set_xlabel('$x$')
        axes[0][0].set_ylabel('$y$')
        axes[0][0].set_aspect('equal')

        axes[0][1].imshow(u[:, :, 1].T.cpu(), **im_kwargs)
        axes[0][1].set_title(f'Original $y$ displacement')
        axes[0][1].set_xlabel('$x$')
        axes[0][1].set_ylabel('$y$')
        axes[0][1].set_aspect('equal')

        axes[1][0].imshow(u_prob[:, :, 0].T.cpu(), **im_kwargs)
        axes[1][0].set_title(f'Most likely $x$ displacement')
        axes[1][0].set_xlabel('$x$')
        axes[1][0].set_ylabel('$y$')
        axes[1][0].set_aspect('equal')

        last = axes[1][1].imshow(u_prob[:, :, 1].T.cpu(), **im_kwargs)
        axes[1][1].set_title(f'Most likely $y$ displacement')
        axes[1][1].set_xlabel('$x$')
        axes[1][1].set_ylabel('$y$')
        axes[1][1].set_aspect('equal')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
        fig.colorbar(last, cax=cbar_ax, label='Displacement')

        output_dir = os.path.join('results/pd2d/ood', run.name + '-' + run.id)
        plt.savefig(
            os.path.join(output_dir, 'compare_prob.png'),
            bbox_inches='tight'
        )
        plt.close()

        fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 8))
        axes[0][0].imshow(u[:, :, 0].T.cpu(), **im_kwargs)
        axes[0][0].set_title(f'Original $x$ displacement')
        axes[0][0].set_xlabel('$x$')
        axes[0][0].set_ylabel('$y$')
        axes[0][0].set_aspect('equal')

        axes[0][1].imshow(u[:, :, 1].T.cpu(), **im_kwargs)
        axes[0][1].set_title(f'Original $y$ displacement')
        axes[0][1].set_xlabel('$x$')
        axes[0][1].set_ylabel('$y$')
        axes[0][1].set_aspect('equal')

        axes[1][0].imshow(u_proj[:, :, 0].T.cpu(), **im_kwargs)
        axes[1][0].set_title(f'Projected $x$ displacement')
        axes[1][0].set_xlabel('$x$')
        axes[1][0].set_ylabel('$y$')
        axes[1][0].set_aspect('equal')

        last = axes[1][1].imshow(u_proj[:, :, 1].T.cpu(), **im_kwargs)
        axes[1][1].set_title(f'Projected $y$ displacement')
        axes[1][1].set_xlabel('$x$')
        axes[1][1].set_ylabel('$y$')
        axes[1][1].set_aspect('equal')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
        fig.colorbar(last, cax=cbar_ax, label='Displacement')

        output_dir = os.path.join('results/pd2d/ood', run.name + '-' + run.id)
        plt.savefig(
            os.path.join(output_dir, 'compare_proj.png'),
            bbox_inches='tight'
        )
        plt.close()

        fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 8))
        axes[0][0].imshow(v[:, :, 0].T.cpu(), **im_kwargs)
        axes[0][0].set_title(f'Final $x$ displacement')
        axes[0][0].set_xlabel('$x$')
        axes[0][0].set_ylabel('$y$')
        axes[0][0].set_aspect('equal')

        axes[0][1].imshow(v[:, :, 1].T.cpu(), **im_kwargs)
        axes[0][1].set_title(f'Final $y$ displacement')
        axes[0][1].set_xlabel('$x$')
        axes[0][1].set_ylabel('$y$')
        axes[0][1].set_aspect('equal')

        axes[1][0].imshow(v_prob[:, :, 0].T.cpu(), **im_kwargs)
        axes[1][0].set_title(f'Pred most likely $x$ displacement')
        axes[1][0].set_xlabel('$x$')
        axes[1][0].set_ylabel('$y$')
        axes[1][0].set_aspect('equal')

        last = axes[1][1].imshow(v_prob[:, :, 1].T.cpu(), **im_kwargs)
        axes[1][1].set_title(f'Pred most likely $y$ displacement')
        axes[1][1].set_xlabel('$x$')
        axes[1][1].set_ylabel('$y$')
        axes[1][1].set_aspect('equal')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
        fig.colorbar(last, cax=cbar_ax, label='Displacement')

        output_dir = os.path.join('results/pd2d/ood', run.name + '-' + run.id)
        plt.savefig(
            os.path.join(output_dir, 'compare_proj.png'),
            bbox_inches='tight'
        )
        plt.close()

        fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(10, 8))
        axes[0][0].imshow(v[:, :, 0].T.cpu(), **im_kwargs)
        axes[0][0].set_title(f'Final $x$ displacement')
        axes[0][0].set_xlabel('$x$')
        axes[0][0].set_ylabel('$y$')
        axes[0][0].set_aspect('equal')

        axes[0][1].imshow(v[:, :, 1].T.cpu(), **im_kwargs)
        axes[0][1].set_title(f'Final $y$ displacement')
        axes[0][1].set_xlabel('$x$')
        axes[0][1].set_ylabel('$y$')
        axes[0][1].set_aspect('equal')

        axes[1][0].imshow(v_proj[:, :, 0].T.cpu(), **im_kwargs)
        axes[1][0].set_title(f'Pred most likely $x$ displacement')
        axes[1][0].set_xlabel('$x$')
        axes[1][0].set_ylabel('$y$')
        axes[1][0].set_aspect('equal')

        last = axes[1][1].imshow(v_proj[:, :, 1].T.cpu(), **im_kwargs)
        axes[1][1].set_title(f'Pred most likely $y$ displacement')
        axes[1][1].set_xlabel('$x$')
        axes[1][1].set_ylabel('$y$')
        axes[1][1].set_aspect('equal')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
        fig.colorbar(last, cax=cbar_ax, label='Displacement')

        output_dir = os.path.join('results/pd2d/ood', run.name + '-' + run.id)
        plt.savefig(
            os.path.join(output_dir, 'compare_proj.png'),
            bbox_inches='tight'
        )
        plt.close()