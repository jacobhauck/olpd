import mlx
import copy
import torch
from torchtnt.utils.flops import FlopTensorDispatchMode


def print_recursive(flop_counts, depth=0):
    for key, value in flop_counts.items():
        if isinstance(value, int):
            print('  ' * depth + f'{key}: {value}')
        else:
            print('  ' * depth + key)
            print_recursive(value, depth=depth + 1)


@mlx.experiment
def count_flops(config, name, group=None):
    model = mlx.create_module(config['model'])
    x = [torch.randn(shape) for shape in config['inputs']]
    with FlopTensorDispatchMode(model) as counter:
        result = model(*x).mean()
        flops_forward = copy.deepcopy(counter.flop_counts)

        counter.reset()
        result.backward()
        flops_backward = copy.deepcopy(counter.flop_counts)

    print(model)

    print('Forward pass:')
    print_recursive(flops_forward)
    print()

    print('Backward pass:')
    print_recursive(flops_backward)
    print()
