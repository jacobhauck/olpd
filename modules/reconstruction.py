import torch
import mlx


def coefficients(basis, u, x, integrator=None):
    """
    Gets coefficients of the best approximation of a function in the given basis
    :param basis: (B, *shape, p, d_out) basis values
    :param u: (B, *shape, d_out) function to approximate
    :param x: (B, *shape, d_in) sample points
    :param integrator: Optional integration module
    :return: (B, p) coefficients of best approximation
    """
    b, p = basis.shape[0], basis.shape[-2]
    if integrator is not None:
        prod = torch.einsum('b...pd,b...d->b...p', basis, u)
        # (B, *shape, p)
        inner_prod = integrator(prod, x)  # (B, p)

        self_prod = torch.einsum('b...pd,b...qd->b...pq', basis, basis)
        # (B, *shape, p, p)
        self_prod = self_prod.flatten(-2)  # (B, *shape, p*p)
        self_inner_prod = integrator(self_prod, x)  # (B, p*p)
        self_inner_prod = self_inner_prod.reshape(b, p, p)  # (B, p, p)

        solution = torch.linalg.solve(self_inner_prod, inner_prod[..., None])
        return solution[..., 0]  # (B, p)
    else:
        a_mat = torch.transpose(basis, -1, -2).reshape(b, -1, p)
        # (B, d, p)
        b_mat = u.reshape(b, -1, 1)  # (B, d, 1)
        lsq = torch.linalg.lstsq(a_mat, b_mat)
        return lsq.solution[..., 0]  # (B, p)


class ReconstructionLoss(torch.nn.Module):
    def __init__(self, squared=True, relative=False, integrator=None):
        super().__init__()
        self.relative = relative
        self.squared = squared
        if integrator is not None:
            self.integrator = mlx.create_module(integrator)
        else:
            self.integrator = None

    def forward(self, basis, u, x):
        """
        :param basis: (B, *shape, p, d_out)
        :param u: (B, *shape, d_out)
        :param x: (B, *shape, d_in)
        :return: Reconstruction loss
        """
        coef = coefficients(basis, u, x, self.integrator)
        # (B, p, 1)
        recon = torch.einsum('b...pd,bp->b...d', basis, coef)
        # (B, *shape, d_out)

        res = torch.sum((recon - u) ** 2, dim=-1, keepdim=True)
        # (B, *shape, 1)

        if self.relative:
            u_norm2 = torch.sum(u**2, dim=-1, keepdim=True)  # (B, *shape, 1)
            if self.integrator is not None:
                loss_sq = self.integrator(res, x) / self.integrator(u_norm2, x)
                # (B, 1)
            else:
                loss_sq = torch.mean(res.flatten(1)) / torch.mean(u_norm2.flatten(1))
                # (B,)
        else:
            if self.integrator is not None:
                loss_sq = self.integrator(res, x)  # (B, 1)
            else:
                loss_sq = torch.mean(res.flatten(1))  # (B,)

        if self.squared:
            return torch.mean(loss_sq)
        else:
            return torch.mean(torch.sqrt(loss_sq))
