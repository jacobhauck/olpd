import mlx
from operatorlearning.data import OLDataset
from operatorlearning.modules import FunctionalL2Loss
from modules.data import NormalizedOLDataset
import torch.utils.data


class Multiband2dTrainer(mlx.training.BaseTrainer):
    loss_fn = None
    metrics_fns = {}
    rel_l2 = None

    def load_datasets(self, config):
        self.loss_fn = mlx.create_module(config['training']['loss_fn'])
        self.loss_fn.to(config['device'])
        self.rel_l2 = FunctionalL2Loss(relative=True, squared=False)
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

    def downsample(self, u, band):
        """
        Downsample to a particular band resolution
        :param u: (B, N1, N2, d_out) function sample values
        :param band: Index of band
        :return: downsampled u
        """
        if self.model.training:
            r = self.config['training']['resolutions'][band]
            step1 = u.shape[1] // r
            step2 = u.shape[2] // r
            return u[:, ::step1, ::step2]
        else:
            return u

    def loss(self, data):
        u, x, v, y = data
        d = self.config['device']
        u, x, v, y = u.to(d), x.to(d), v.to(d), y.to(d)

        v_dec, y_dec = self.model.decomposition(v, y)

        losses = {'objective': 0.0}
        pred = []
        v_ds, y_ds = [], []
        for i in self.config['training']['bands']:
            u_d, x_d = self.downsample(u, i), self.downsample(x, i)
            v_d, y_d = self.downsample(v_dec[:, i], i), self.downsample(y_dec[:, i], i)

            v_pred = self.model.predict_band(u_d, x_d, y_d, band=i)
            pred.append(v_pred)
            v_ds.append(v_d)
            y_ds.append(y_d)
            loss = self.loss_fn(v_pred, v_d, y_d)
            w = self.config['training']['band_weights'][i]
            losses[f'band_{i}'] = loss
            losses['objective'] = w * loss + losses['objective']

        return (pred, v_ds, y_ds), losses

    def metrics(self, prediction, data):
        _, _, v, y = data
        v, y = v.to(self.config['device']), y.to(self.config['device'])
        v_pred, v_d, y_d = prediction

        metrics = {}
        for name, fn in self.metrics_fns.items():
            if not self.model.training:
                pred_recon = self.model.decomposition.recompose(torch.stack(v_pred, dim=1))
                metrics[f'{name}_rec'] = fn(pred_recon, v, y)

            for i in self.config['training']['bands']:
                metrics[f'{name}_{i}'] = fn(v_pred[i], v_d[i], y_d[i])

        return metrics


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
