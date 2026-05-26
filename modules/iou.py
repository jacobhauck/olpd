import torch


class CrackIoU2d(torch.nn.Module):
    def __init__(self, damage_thresh):
        super().__init__()
        self.damage_thresh = damage_thresh

    # noinspection PyTypeChecker
    def forward(self, prediction, target):
        """
        :param prediction: (B, *shape, 1)
        :param target: (B, *shape, 1)
        :return: average IoU between prediction and target
        """
        b = prediction.shape[0]
        prediction_t = (prediction.reshape(b, -1) > self.damage_thresh)
        target_t = (target.reshape(b, -1) > self.damage_thresh)
        intersection = torch.sum(prediction_t & target_t, dim=1)
        union = torch.sum(prediction_t | target_t, dim=1)
        return torch.mean(intersection / union)
