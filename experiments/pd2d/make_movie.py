import mlx
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from operatorlearning.data import OLDataset
import set_fonts


@mlx.experiment
def make_movie(config, name, group=None):
    dataset = OLDataset(config['movie_dataset'])

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_title('Final $x$ displacement')
    u, _, v, _ = dataset[0]
    im_kwargs = {
        'vmin': min(float(u.min()), float(v.min())),
        'vmax': max(float(u.max()), float(v.max())),
        'cmap': 'seismic',
        'extent': (config['xo'], config['xn'], config['yo'], config['yn']),
        'origin': 'lower'
    }
    im = ax.imshow(u[:, :, 0].T, **im_kwargs)
    fig.tight_layout()

    def update(frame):
        _, _, v_, _ = dataset[frame]
        im.update({'data': v_[:, :, 0].T})
        return [im]

    output_dir = os.path.join('results', name)
    os.makedirs(output_dir, exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=range(32), blit=True)
    ani.save(os.path.join(output_dir, 'movie.gif'), writer="pillow", fps=6)

    print(f'Saved movie in {output_dir}')
