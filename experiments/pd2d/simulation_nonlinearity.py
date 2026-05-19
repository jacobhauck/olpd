import mlx
from operatorlearning.data import OLDataset
from operatorlearning.modules import FunctionalL2Loss


@mlx.experiment
def run_experiment(config, *_, **__):
    rel_l2 = FunctionalL2Loss(relative=True, squared=False)
    errors = {}
    base_dataset = OLDataset(config['base_dataset'], stream_uv=False)
    for scale, dataset in config['datasets'].items():
        print('Calculating error for scale:', scale)
        total = 0
        for test, base in zip(OLDataset(dataset, stream_uv=False), base_dataset):
            v_test = test[2] / float(scale)
            v_base = base[2]
            total += float(rel_l2(v_test[None], v_base[None]))
        errors[scale] = 100 * total / len(base_dataset)

    print(errors)
