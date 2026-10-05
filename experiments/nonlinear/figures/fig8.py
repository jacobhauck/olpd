import mlx
import matplotlib.pyplot as plt
from operatorlearning.data import OLDataset, OLDatasetLibrary
import set_fonts


@mlx.experiment
def plot_figure(config, name, group=None):
    fig, axes = plt.subplots(1, 3, figsize=config['figure_size'])

    lib = OLDatasetLibrary('nonlinear')
    data_linear = OLDataset(lib.dataset_path(config['split'], config['linear_id']))
    data_nonlinear = OLDataset(lib.dataset_path(config['split'], config['nonlinear_id']))
    u_l, _, v_l, _ = data_linear[config['sample']]
    u_n, _, v_n, _ = data_nonlinear[config['sample']]

    im_kwargs = {
        'cmap': 'seismic',
        'origin': 'lower'
    }

    max_val = u_l.abs().max().item()
    axes[0].imshow(u_l[:, :, 0].T, **im_kwargs, vmin=-max_val, vmax=max_val)
    axes[0].set_axis_off()
    axes[0].set_title('$u$')

    max_val = v_l.abs().max().item()
    axes[1].imshow(v_l[:, :, 0].T, **im_kwargs, vmin=-max_val, vmax=max_val)
    axes[1].set_axis_off()
    axes[1].set_title(f'$v$ (linear, $a = 0$)')

    max_val = v_n.abs().max().item()
    axes[2].imshow(v_n[:, :, 0].T, **im_kwargs, vmin=-max_val, vmax=max_val)
    axes[2].set_axis_off()
    axes[2].set_title(f'$v$ (\\textbf{{nonlinear}}, $a = {lib[config["nonlinear_id"]]["a"]}$)')

    mlx.show_and_save(fig, 'sample', config, name)
