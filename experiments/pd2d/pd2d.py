import mlx
from operatorlearning.data import OLDataset
import torch.utils.data

from modules.data import NormalizedOLDataset


class PD2DTrainer(mlx.training.BaseTrainer):
    loss_fn = None
    metrics_fns = {}

    def load_datasets(self, config):
        self.loss_fn = mlx.create_module(config['training']['loss_fn'])
        self.loss_fn.to(config['device'])
        self.metrics_fns = {
            name: mlx.create_module(conf).to(config['device'])
            for name, conf in config.get('metrics', {}).items()
        }

        train_dataset = OLDataset(**config['data']['train'])
        test_dataset = OLDataset(**config['data']['test'])

        if config['training'].get('normalize', False):
            train_dataset = NormalizedOLDataset(train_dataset)
            test_dataset = NormalizedOLDataset(
                test_dataset,
                u_mean=train_dataset.u_mean,
                u_std=train_dataset.u_std,
                v_mean=train_dataset.v_mean,
                v_std=train_dataset.v_std
            )

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
        v_pred = self.apply_model(u, x, y)
        loss = self.loss_fn(v_pred, v)
        return v_pred, {'objective': loss}

    def metrics(self, prediction, data):
        _, _, v, _ = data
        v = v.to(self.config['device'])
        return {name: fn(prediction, v) for name, fn in self.metrics_fns.items()}


class PD2DTraining(mlx.WandBExperiment):
    def wandb_run(self, config, run):
        save_interval = config['training'].get('save_interval', 600)
        log_interval = config['training'].get('log_interval', 3)
        trainer = PD2DTrainer(
            config, run,
            save_interval=save_interval,
            log_interval=log_interval
        )

        # Fit PCA bases if necessary
        if 'pcanet' in config['model']['name'].lower():
            # sample = (u, x, v, y)
            # noinspection PyTypeChecker
            uv_map = map(lambda sample: (sample[0], sample[2]), trainer.datasets['train'])
            _, _x, _, _y = trainer.datasets['train'][0]
            print('Fitting PCA bases')
            trainer.model.fit_pca(uv_map, _x, _y)

        # Handle model interface compatibility
        if 'fno' in config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model(u)
        elif 'gnot' in config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model([(u, x)], y)
        elif 'pcanet' in config['model']['name'].lower():
            trainer.apply_model = lambda u, x, y: trainer.model(u)

        trainer.train(epochs=config['training']['epochs'])
        losses, metrics = trainer.evaluate(('train', 'test'))

        for dataset, dataset_losses in losses.items():
            print(f'===== Loss for dataset: "{dataset}" =====')
            for loss_name, loss in dataset_losses.items():
                print(f'    === Loss: {loss_name} ===')
                print(f'    Mean: {loss.mean().item():.05f}')
                print(f'    Median: {loss.median().item():.05f}')
                print(f'    Std.: {loss.std().item():.05f}')

        for dataset, dataset_metrics in metrics.items():
            print(f'===== Metric for dataset: "{dataset}" =====')
            for metric_name, metric in dataset_metrics.items():
                print(f'    === Loss: {metric_name} ===')
                print(f'    Mean: {metric.mean().item():.05f}')
                print(f'    Median: {metric.median().item():.05f}')
                print(f'    Std.: {metric.std().item():.05f}')
