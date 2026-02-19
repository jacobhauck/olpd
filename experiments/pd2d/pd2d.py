import mlx
from operatorlearning.data import OLDataset
import torch.utils.data


class PD2DTrainer(mlx.training.BaseTrainer):
    loss_fn = None
    metrics_fns = {}

    def load_datasets(self, config):
        self.loss_fn = mlx.create_module(config['training']['loss_fn'])
        self.metrics_fns = {conf['name']: mlx.create_module(conf) for conf in config.get('metrics', ())}

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
        losses, _ = trainer.evaluate(('train', 'test'))

        for dataset, dataset_losses in losses.items():
            print(f'===== Loss for dataset: "{dataset}" =====')
            for loss_name, losses in dataset_losses.items():
                print(f'    === Loss: {loss_name} ===')
                print(f'    Mean: {losses.mean().item():.05f}')
                print(f'    Median: {losses.median().item():.05f}')
                print(f'    Std.: {losses.std().item():.05f}')
