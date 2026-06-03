import mlx
import modules.data
from operatorlearning.data import OLDataset


@mlx.experiment
def subsampling_error(config, name, group=None):
    dataset = OLDataset(config['dataset'])
    temp_file = config['dataset'] + '.temp.ol.h5'
    modules.data.pd2d_subsample_dataset(config['dataset'], temp_file, config['nx'], config['ny'])
    downsampled = OLDataset(temp_file)

