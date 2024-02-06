import torch
from torch import nn, Tensor
from torch.nn import functional as F

__all__ = ['BCEWithLogitsLoss','BinaryCrossEntropy','CrossEntropy', 'MSELoss', 'BCELoss', 'NLLLoss', 'KLDivLoss', 'HingeLoss', 'SmoothL1Loss']

class BCEWithLogitsLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
       return torch.nn.BCEWithLogitsLoss()(input, target.float())

class BinaryCrossEntropy(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return F.binary_cross_entropy(input, target)

class CrossEntropy(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return F.cross_entropy(input, target)

class MSELoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return F.mse_loss(input, target)


class BCELoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return F.binary_cross_entropy_with_logits(input, target)


class NLLLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return F.nll_loss(input, target)


class KLDivLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return F.kl_div(input, target)


class HingeLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return F.hinge_embedding_loss(input, target)


class SmoothL1Loss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return F.smooth_l1_loss(input, target)


def get_loss(loss_fn_name: str = 'CrossEntropy'):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    return eval(loss_fn_name)()#return torch.nn.CrossEntropyLoss()
