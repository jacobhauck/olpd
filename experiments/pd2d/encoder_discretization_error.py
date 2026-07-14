import mlx
import torch
from .pd2d import PD2DTrainer
from operatorlearning.data import OLDataset


@mlx.experiment
def run_experiment(config, name, group=None):
    run = mlx.load_run(config['run_id'])
    run.config['device'] = config['device']
    trainer = PD2DTrainer(run.config, run)

    datasets = {name: OLDataset(file) for name, file in config['datasets'].items()}
    errors = {name: {} for name in config['datasets']}
    rel_errors = {name: {} for name in config['datasets']}

    d = config['device']

    for name, dataset in datasets.items():
        for other_name, other_dataset in datasets.items():
            if name in errors[other_name]:
                continue

            if name == other_name:
                errors[name][other_name] = [0.0] * len(dataset)
                rel_errors[name][other_name] = [0.0] * len(dataset)
                continue
            else:
                print(f'Computing error between {name} and {other_name}')
                errors[name][other_name] = []
                rel_errors[name][other_name] = []

            for i in range(len(dataset)):
                u, x, v, y = dataset[i]
                u, x, v, y = u.to(d), x.to(d), v.to(d), y.to(d)

                encoder_basis = trainer.model.encoder_net(x[None])  # (1, *in_shape, p, u_d_out)
                prod = torch.einsum('B...d,B...pd->B...p', u[None], encoder_basis)
                # (1, *in_shape, p)

                trainer.on_evaluation_start()
                z = trainer.model.integrator(prod, x[None])  # (1, p)

                u2, x2, v2, y2 = other_dataset[i]
                u2, x2, v2, y2 = u2.to(d), x2.to(d), v2.to(d), y2.to(d)
                encoder_basis = trainer.model.encoder_net(x2[None])  # (1, *in_shape2, p, u_d_out)
                prod = torch.einsum('B...d,B...pd->B...p', u2[None], encoder_basis)
                # (1, *in_shape2, p)

                trainer.on_evaluation_start()
                z2 = trainer.model.integrator(prod, x2[None])  # (1, p)

                errors[name][other_name].append(torch.sum((z - z2)**2).item())
                rel_errors[name][other_name].append((torch.sum((z - z2)**2) / torch.sum(z**2)).item())

    for name in datasets:
        for other_name in datasets:
            if other_name not in errors[name]:
                continue
            e = torch.tensor(errors[name][other_name])
            r = torch.tensor(rel_errors[name][other_name])
            print(name, other_name, e.mean().item(), (e**.5).mean().item(), r.mean().item(), (r**.5).mean().item())
