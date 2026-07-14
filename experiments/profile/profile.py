import mlx
import torch
import time


@mlx.experiment
def measure_time(config, name, group=None):
    model = mlx.create_module(config['model'])
    model.to(config['device'])
    x = [torch.randn(shape, device=config['device']) for shape in config['inputs']]

    # One warm-up round
    result = model(*x).mean()
    result.backward()

    t_forward = []
    t_backward = []
    for i in range(config['iterations']):
        t = time.time()
        result = model(*x).mean()
        t_forward.append(time.time() - t)
        t = time.time()
        result.backward()
        t_backward.append(time.time() - t)
    t_forward, t_backward = map(torch.tensor, (t_forward, t_backward))
    print(f'Average forward time: {t_forward.mean()} +/- {t_forward.std()}s')
    print(f'Average backward time: {t_backward.mean()}s +/- {t_backward.std()}s')
