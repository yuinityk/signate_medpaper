import numpy as np
from sklearn.metrics import fbeta_score

BETA_FOR_FBETA = 7


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self, thr):
        self.beta = BETA_FOR_FBETA
        self.thr = thr
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())

        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.where(y_pred > self.thr, 1, 0)
        self.y_pred.extend(y_pred.tolist())

    @property
    def avg(self):
        self.score = fbeta_score(self.y_true, self.y_pred, beta=self.beta)
        return {
            'FBeta': self.score
        }

