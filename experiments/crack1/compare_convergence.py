import mlx
import matplotlib.pyplot as plt
import os
from operatorlearning.data import OLDataset
from operatorlearning.modules import FunctionalL2Loss


@mlx.experiment
def run_experiment(config, name, group=None):
    l2_loss = FunctionalL2Loss(relative=False, squared=False)

    dt = config['settings']['dt over dt_0']
    fig, axes = plt.subplots(
        len(dt), len(config['datasets']),
        figsize=(9, 1.5*len(dt))
    )

    im_kwargs = {
        'vmin': 0,
        'vmax': 0.8,
        'cmap': 'plasma',
        'origin': 'lower'
    }

    i_dataset = 0
    for dataset_name in config['datasets']:
        dataset = OLDataset(config['datasets'][dataset_name])

        print('Dataset', dataset_name)
        print(f'dt/d_0      Error from ref')
        for i, dt_i in enumerate(dt):
            if i < len(dt) - 1:
                err = l2_loss(dataset[i][2], dataset[-1][2])
                print(str(dt_i) + ' ' * (12 - len(str(dt_i))) + f'{err}')
        print()
        for i, dt_i in enumerate(dt):
            v = dataset[i][2]
            v_crop = v[config['crop']['x'][0] : config['crop']['x'][1], config['crop']['y'][0] : config['crop']['y'][1]]
            axes[i][i_dataset].imshow(v_crop[:, :, 0].T, **im_kwargs)

            if i_dataset == 0:
                axes[i][i_dataset].set_ylabel(dt_i)
                axes[i][i_dataset].set_xticks([])
                axes[i][i_dataset].set_yticks([])
            else:
                axes[i][i_dataset].set_axis_off()

        axes[0][i_dataset].set_title(dataset_name)

        i_dataset += 1

    if config.get('show', False):
        plt.show()

    save_folder = os.path.join('results', name)
    os.makedirs(save_folder, exist_ok=True)
    file_format = config.get('format', 'png')
    file_name = os.path.join(save_folder, f'comparison.{file_format}')
    fig.savefig(file_name, bbox_inches='tight')
    print(f'Saved figure at {file_name}')
