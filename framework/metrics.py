import torch
from torch import Tensor
from typing import Tuple,List
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score

class Metrics:
    def __init__(self, num_classes: int, device) -> None:
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)
        self.all_preds = []
        self.all_targets = []

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        self.hist += torch.histc(target * self.num_classes + pred, bins=self.num_classes**2, min=0, max=self.num_classes**2-1).reshape(self.num_classes, self.num_classes)
        self.all_preds.extend(pred.tolist())
        self.all_targets.extend(target.tolist())

    def compute_acc(self):
        acc = sum([self.all_preds[i] == self.all_targets[i] for i in range(len(self.all_preds))]) / len(self.all_preds)
        return acc

    def compute_f1(self):
        f1 = f1_score(self.all_targets, self.all_preds, average='macro')
        return f1

    def compute_rec(self):
        rec = recall_score(self.all_targets, self.all_preds, average='macro')
        return rec

    def compute_prec(self):
        prec = precision_score(self.all_targets, self.all_preds, average='macro')
        return prec

    def compute_roc_auc(self):
        roc_auc = roc_auc_score(self.all_targets, self.all_preds)
        return roc_auc

    def compute_pr_auc(self):
        pr_auc = average_precision_score(self.all_targets, self.all_preds)
        return pr_auc


    '''
    #keep = target != self.ignore_label
        #self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)
    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)
    '''