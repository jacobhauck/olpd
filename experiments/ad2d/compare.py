import mlx
import matplotlib.pyplot as plt
import set_fonts
import os
from operatorlearning.data import OLDataset


@mlx.experiment
def compare(config, name, group=None):
    parabolic = OLDataset(config['parabolic_dataset'])
    hyperbolic = OLDataset(config['hyperbolic_dataset'])

    output_folder = os.path.join('results', name)
    os.makedirs(output_folder, exist_ok=True)
    im_kwargs = {
        'origin': 'lower',
        'cmap': 'seismic'
    }
    for i in mlx.subset_indices(config, parabolic):
        u_p, x_p, v_p, y_p = parabolic[i]
        u_h, x_h, v_h, y_h = hyperbolic[i]

        fig, axes = plt.subplots(1, 3, sharey=True, figsize=config['figure_size'])
        big_value = u_p.abs().max().item()
        im_kwargs['vmin'] = -big_value
        im_kwargs['vmax'] = big_value

        axes[0].imshow(u_h[:, :, 0].T, **im_kwargs)
        axes[0].set_axis_off()
        axes[0].set_title('Input (initial state)')

        axes[1].imshow(v_h[:, :, 0].T, **im_kwargs)
        axes[1].set_axis_off()
        axes[1].set_title('Hyperbolic (final state)')

        big_value = v_p.abs().max().item()
        im_kwargs['vmin'] = -big_value
        im_kwargs['vmax'] = big_value
        axes[2].imshow(v_p[:, :, 0].T, **im_kwargs)
        axes[2].set_axis_off()
        axes[2].set_title('Parabolic (final state)')

        if config['show']:
            plt.show()

        fig.savefig(os.path.join(output_folder, f'{i}.{config.get("format", "png")}'), bbox_inches='tight')

        plt.close(fig)
