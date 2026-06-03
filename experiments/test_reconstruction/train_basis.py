import mlx
import torch.utils.data

from modules.reconstruction import ReconstructionLoss
from operatorlearning.data import OLDataset


class BasisTrainer(mlx.training.BaseTrainer):
    loss_fn = None

    def loss(self, data):
        if self.config.get('target_dist', 'input') == 'input':
            u, x, _, _ = data
            d = self.config['device']
            u, x = u.to(d), x.to(d)
        else:
            _, _, v, y = data
            d = self.config['device']
            u, x = v.to(d), y.to(d)


        basis_val = self.model(x)
        return basis_val, {'objective': self.loss_fn(basis_val, u, x)}

    def load_datasets(self, config):
        if config['use_integrator']:
            self.loss_fn = ReconstructionLoss(config['integrator'])
        else:
            self.loss_fn = ReconstructionLoss()

        datasets = {
            name: OLDataset(**params)
            for name, params in config['data'].items()
        }

        data_loaders = {}
        for name, dataset in datasets.items():
            if name == 'train':
                data_loaders[name] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config['training']['batch_size'],
                    shuffle=True,
                    drop_last=False
                )
            else:
                data_loaders[name] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1
                )

        return datasets, data_loaders


@mlx.wandb_experiment
def train_basis(config, run):
    trainer = BasisTrainer(config, run, log_interval=3, save_interval=600)
    trainer.train(epochs=config['training']['epochs'])
