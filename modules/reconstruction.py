import torch
import mlx


class ReconstructionLoss(torch.nn.Module):
    def __init__(self, integrator=None):
        super().__init__()
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
        b, p, d_out = basis.shape[0], basis.shape[-2], basis.shape[-1]
        if self.integrator is not None:
            prod = torch.einsum('b...pd,b...d->b...p', basis, u)
            # (B, *shape, p)
            inner_prod = self.integrator(prod, x)  # (B, p)

            self_prod = torch.einsum('b...pd,b...qd->b...pq', basis, basis)
            # (B, *shape, p, p)
            self_prod = self_prod.flatten(-2)  # (B, *shape, p*p)
            self_inner_prod = self.integrator(self_prod, x)  # (B, p*p)
            self_inner_prod = self_inner_prod.reshape(b, p, p)  # (B, p, p)

            solution = torch.linalg.solve(self_inner_prod, inner_prod[..., None])
            # (B, p, 1)
            recon = torch.einsum('b...pd,bp->b...d', basis, solution[..., 0])
            # (B, *shape, d_out)

            norm2 = torch.sum((recon - u) ** 2, dim=-1, keepdim=True)  # (B, *shape, 1)
            return torch.mean(self.integrator(norm2, x))
        else:
            a_mat = torch.transpose(basis, -1, -2).reshape(b, -1, p)
            # (B, d, p)
            b_mat = u.reshape(b, -1, 1)  # (B, d, 1)
            lsq = torch.linalg.lstsq(a_mat, b_mat)
            z = lsq.solution[..., 0]  # (B, p)
            recon = torch.einsum('b...pd,bp->b...d', basis, z)
            # (B, *shape, d_out)
            res = torch.mean(((recon - u)**2).reshape(b, -1), dim=1) * d_out
            # (B,)
            return torch.mean(res)
