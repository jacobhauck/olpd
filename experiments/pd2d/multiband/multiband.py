import mlx
from operatorlearning.data import OLDataset
import torch.utils.data


class Multiband2dTrainer(mlx.training.BaseTrainer):
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

    def downsample(self, v_d, y_d):
        """
        :param v_d: (B, num_steps, N1, N2, 2) decomposed bands
        :param y_d: (B, num_steps, N1, N2, 2) expanded sample points
        :return: tuple (v_dd, y_dd), where v_dd is a list of
            (B, N1_i, N2_i, v_d_out) tensors of downsampled band values and
            y_dd is a list of (B, N1_i, N2_i, 2) of downsampled points
        """
        result_v = []
        result_y = []
        for i in range(y_d.shape[1]):
            step1 = y_d.shape[2] // self.config['training']['resolutions'][i]
            step2 = y_d.shape[3] // self.config['training']['resolutions'][i]
            result_v.append(v_d[:, i, ::step1, ::step2])
            result_y.append(y_d[:, i, ::step1, ::step2])

        return result_v, result_y

    def loss(self, data):
        u, x, v, y = data
        d = self.config['device']
        u, x, v, y = u.to(d), x.to(d), v.to(d), y.to(d)

        if self.model.training:
            v_d, y_d = self.model.decomposition(v, y)
            v_dd, y_dd = self.downsample(v_d, y_d)
            bands = self.model(u, x, y_dd, bands=self.config['training']['bands'])
            v_dd = [v_dd[i] for i in self.config['training']['bands']]
            y_dd = [y_dd[i] for i in self.config['training']['bands']]
            band_losses, loss = self.loss_fn(bands, v_dd, y_dd)

            loss_dict = {'objective': loss}
            for i in range(len(band_losses)):
                loss_dict[f'band_{i}'] = band_losses[i]
            return bands, loss_dict
        else:
            v_pred = self.model(u, x, y)
            return v_pred, {}

    def metrics(self, prediction, data):
        if not isinstance(prediction, torch.Tensor):
            return {}

        _, _, v, y = data
        v, y = v.to(self.config['device']), y.to(self.config['device'])
        return {name: fn(prediction, v, y) for name, fn in self.metrics_fns.items()}


class MultibandExperiment(mlx.WandBExperiment):
    def wandb_run(self, config, run):
        save_interval = config['training'].get('save_interval', 600)
        log_interval = config['training'].get('log_interval', 3)
        trainer = Multiband2dTrainer(
            config, run,
            save_interval=save_interval,
            log_interval=log_interval
        )

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
