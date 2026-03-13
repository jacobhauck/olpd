import mlx
from operatorlearning.data import OLDataset
import torch.utils.data


class Multiband2dTrainer(mlx.training.BaseTrainer):
    loss_fn = None
    metrics_fns = {}
    decomposition = None

    def load_datasets(self, config):
        self.loss_fn = mlx.create_module(config['training']['loss_fn'])
        self.loss_fn.to(config['device'])
        self.metrics_fns = [
            mlx.create_module(conf).to(config['device'])
            for conf in config.get('metrics', ())
        ]
        self.decomposition = mlx.create_module(config['multiband']['decomposition'])

        train_dataset = OLDataset(**config['data']['train'])
        test_dataset = OLDataset(**config['data']['test'])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            drop_last=True
        )

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

        return (
            {'train': train_dataset, 'test': test_dataset},
            {'train': train_loader, 'test': test_loader}
        )

    def apply_model(self, u, x, y):
        # Default implementation; works for MFEAR
        return self.model(u, x_in=x, x_out=y)

    def loss(self, data):
        u, x, v, y = data
        d = self.config['device']
        u, x, v, y = u.to(d), x.to(d), v.to(d), y.to(d)

        v_d, y_d = self.decomposition(v, y)
        v_pred = self.apply_model(u, x, y)
        loss = self.loss_fn(v_pred, v_d[:, -1], x=y_d[:, -1])

        return v_pred, {'objective': loss}

    def metrics(self, prediction, data):
        _, _, v, _ = data
        v = v.to(self.config['device'])
        return {repr(fn): fn(prediction, v) for fn in self.metrics_fns}
