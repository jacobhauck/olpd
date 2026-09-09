import mlx
import matplotlib.pyplot as plt
from operatorlearning.data import OLDatasetLibrary, OLDataset
import set_fonts


@mlx.experiment
def make_plot(config, name, group=None):
    lib = OLDatasetLibrary('ad2d')
    datasets = [
        OLDataset(lib.dataset_path(config['split'], dataset_id, config.get('resolution')))
        for dataset_id in config['datasets']
    ]
    fig_size = ((1 + len(datasets)) * config['fig_size'] + 1, config['fig_size'] + 1)
    fig, axes = plt.subplots(1, 1 + len(datasets), figsize=fig_size)

    u, _, v, _ = datasets[0][config['index']]
    max_val = float(u.abs().max())

    im_kwargs = {
        'origin': 'lower',
        'vmin': -max_val,
        'vmax': max_val,
        'cmap': 'seismic'
    }

    axes[0].imshow(u[:, :, 0].T, **im_kwargs)
    axes[0].set_axis_off()
    axes[0].set_title('Initial condition')

    for i, dataset in enumerate(datasets):
        _, _, v, _ = dataset[config['index']]
        axes[i + 1].imshow(v[:, :, 0].T, **im_kwargs)
        axes[i + 1].set_axis_off()
        axes[i + 1].set_title(f'$k = {int(round(lib[config["datasets"][i]]["k"] / 1e-5)):d}$')

    fig.tight_layout()

    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bbox0 = get_bbox(axes[0])
    bbox1 = get_bbox(axes[1])
    x_middle = (bbox0.x1 + bbox1.x0) / 2

    line = plt.Line2D(
        [x_middle, x_middle], [0.1, 0.9],
        transform=fig.transFigure, color='black', linestyle='--'
    )
    fig.add_artist(line)

    if config.get('show', False):
        plt.show()

    mlx.show_and_save(fig, 'figure_6', config, name)
