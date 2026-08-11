import mlx


@mlx.experiment
def count_params(config, name, group=None):
    if 'run_id' in config:
        model = mlx.create_module(mlx.load_run(config['run_id']).config['model'])
    else:
        model = mlx.create_module(config['model'])
    num_params = 0

    for param in model.parameters():
        num_params += param.numel()

    print('Model')
    print(f'Total parameters = {num_params}')
    print(model)
    print()
